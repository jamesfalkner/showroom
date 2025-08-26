#!/usr/bin/env python3
"""
Platform Engineering Workshop Chatbot Backend with Embedded RAG
FastAPI service with in-memory vector search and workshop content ingestion
"""

import os
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator
from pathlib import Path
import re

import httpx
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from fastmcp import Client

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_value = getattr(logging, log_level, logging.INFO)
logging.basicConfig(level=log_level_value)
logger = logging.getLogger(__name__)

# Configuration from environment variables
class Config:
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_API_URL = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
    STATIC_DIR = os.getenv("STATIC_DIR", "/app/static")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))


config = Config()

# Pydantic models
class ConversationMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    message: str = Field(..., description="Current user message")
    conversation_history: List[ConversationMessage] = Field(default_factory=list, description="Previous conversation messages")
    include_mcp: bool = Field(default=True, description="Whether to include MCP tools")
    page_context: Optional[str] = Field(default=None, description="Current page title or context for focused assistance")

# FastAPI app
app = FastAPI(
    title="Workshop Chatbot Backend - Embedded RAG",
    description="Platform Engineering Workshop AI Assistant with embedded RAG and MCP integration",
    version="2.0.0-embedded-rag"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleRAGEngine:
    """Simple in-memory RAG implementation using TF-IDF and cosine similarity"""

    def __init__(self):
        self.documents = []
        self.document_metadata = []
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.document_vectors = None
        self.is_initialized = False

    def ingest_documents(self, documents: List[Dict]):
        """Ingest documents and create vector index"""
        logger.info(f"=== RAG CONTENT INGESTION START ===")
        logger.info(f"Ingesting {len(documents)} documents...")

        self.documents = []
        self.document_metadata = []

        for i, doc in enumerate(documents):
            # Log each file being processed
            file_path = doc.get('file_path', 'Unknown')
            title = doc.get('title', 'Untitled')
            content_type = doc.get('content_type', 'page')
            logger.info(f"  [{i+1}/{len(documents)}] Processing: {file_path} - {title} ({content_type})")
            
            # Extract clean text content
            content = self._clean_text(doc.get('content', ''))
            original_length = len(doc.get('content', ''))
            cleaned_length = len(content.strip())
            
            if len(content.strip()) > 50:  # Only include substantial content
                self.documents.append(content)
                self.document_metadata.append({
                    'title': title,
                    'module': doc.get('module', 'General'),
                    'file_path': file_path,
                    'content_type': content_type,
                    'length': len(content)
                })
                logger.info(f"    ✓ Added to RAG index (original: {original_length} chars, cleaned: {cleaned_length} chars)")
            else:
                logger.info(f"    ✗ Skipped (too short: {cleaned_length} chars)")

        if self.documents:
            # Create TF-IDF vectors
            logger.info(f"Creating TF-IDF vectors for {len(self.documents)} documents...")
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
            self.is_initialized = True
            logger.info(f"✓ RAG engine initialized with {len(self.documents)} documents")
            logger.info(f"=== RAG CONTENT INGESTION COMPLETE ===")
        else:
            logger.warning("No documents found for RAG initialization")
            logger.info(f"=== RAG CONTENT INGESTION COMPLETE (NO DOCUMENTS) ===")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""

        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', ' ', text)

        return text.strip()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents using cosine similarity"""
        if not self.is_initialized or not self.documents:
            logger.warning(f"RAG SEARCH: Engine not initialized or no documents available")
            return []

        try:
            # Transform query using the same vectorizer
            cleaned_query = self._clean_text(query)
            logger.debug(f"RAG SEARCH: Original query: '{query}' -> Cleaned: '{cleaned_query}'")

            query_vector = self.vectorizer.transform([cleaned_query])

            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()

            # Log similarity statistics
            max_sim = np.max(similarities)
            min_sim = np.min(similarities)
            mean_sim = np.mean(similarities)
            logger.debug(f"RAG SEARCH: Similarity stats - Max: {max_sim:.3f}, Min: {min_sim:.3f}, Mean: {mean_sim:.3f}")

            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            threshold = 0.1
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > threshold:
                    doc_metadata = self.document_metadata[idx]
                    doc_content = self.documents[idx]
                    logger.info(f"RAG MATCH: {doc_metadata['title']} (similarity: {similarity:.3f})")
                    logger.info(f"  File: {doc_metadata['file_path']}")
                    logger.info(f"  Content preview: {doc_content[:200]}...")
                    
                    results.append({
                        'content': doc_content,
                        'metadata': doc_metadata,
                        'similarity': float(similarity)
                    })
                else:
                    logger.debug(f"RAG SEARCH: Skipping document {idx} (similarity {similarity:.3f} below threshold {threshold})")

            logger.debug(f"RAG SEARCH: Returning {len(results)} documents above threshold")
            return results

        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return []

class FastMCPManager:
    """Simplified MCP integration using FastMCP client with multi-server support"""
    
    def __init__(self, mcp_config: Dict):
        self.mcp_config = mcp_config
        self.client = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize FastMCP client with multi-server configuration"""
        if self._initialized:
            return
            
        logger.info(f"=== INITIALIZING FASTMCP CLIENT ===")
        
        if "mcpServers" not in self.mcp_config:
            logger.error("No mcpServers configuration found")
            return
            
        logger.info(f"Using FastMCP multi-server configuration")
        logger.info(f"Config: {json.dumps(self.mcp_config, indent=2)}")
        
        try:
            # Create FastMCP client with the full config
            self.client = Client(self.mcp_config)
            logger.info(f"✓ FastMCP multi-server client created")
            
        except Exception as e:
            logger.error(f"Failed to create FastMCP multi-server client: {e}")
            return
        
        self._initialized = True
        logger.info(f"FastMCP initialization complete")
        
        
    async def get_all_tools(self) -> List[Dict]:
        """Get tools from all configured MCP servers using FastMCP, filtered by allowed_tools"""
        await self.initialize()
        
        if not self.client:
            logger.warning("FastMCP client not initialized")
            return []
        
        all_tools = []
        logger.info("=== FASTMCP TOOLS DISCOVERY ===")
        
        try:
            async with self.client as client:
                # Get available tools from all servers
                tools = await client.list_tools()
                logger.info(f"Found {len(tools)} total tools across all servers")
                
                # Build allowed tools set from configuration
                allowed_tools_set = set()
                if "mcpServers" in self.mcp_config:
                    for server_name, server_config in self.mcp_config["mcpServers"].items():
                        if "allowed_tools" in server_config:
                            server_allowed = server_config["allowed_tools"]
                            allowed_tools_set.update(server_allowed)
                            logger.info(f"Server '{server_name}' allows tools: {server_allowed}")
                
                # If no allowed_tools configured anywhere, allow all tools
                if not allowed_tools_set:
                    logger.info("No allowed_tools restrictions found - allowing all tools")
                    filter_tools = False
                else:
                    logger.info(f"Tool filtering enabled - allowing only: {list(allowed_tools_set)}")
                    filter_tools = True
                
                # Convert Tool objects to dictionaries and filter
                filtered_count = 0
                for tool in tools:
                    # Check if tool should be included
                    # FastMCP prefixes tool names with server name, so check both prefixed and unprefixed
                    tool_name_parts = tool.name.split('_', 1)  # Split on first underscore
                    unprefixed_name = tool_name_parts[1] if len(tool_name_parts) > 1 else tool.name
                    
                    is_allowed = (
                        not filter_tools or 
                        tool.name in allowed_tools_set or  # Check full prefixed name
                        unprefixed_name in allowed_tools_set  # Check unprefixed name
                    )
                    
                    if not is_allowed:
                        logger.debug(f"  - {tool.name} (unprefixed: {unprefixed_name}): FILTERED OUT (not in allowed_tools)")
                        filtered_count += 1
                        continue
                    
                    # Convert Tool object to dictionary
                    tool_dict = {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema.model_dump() if hasattr(tool.inputSchema, 'model_dump') else tool.inputSchema,
                        "_transport": "fastmcp"
                    }
                    
                    logger.info(f"  - {tool_dict['name']} (unprefixed: {unprefixed_name}): ALLOWED - {tool_dict['description']}")
                    all_tools.append(tool_dict)
                    
        except Exception as e:
            logger.error(f"Error getting tools from FastMCP: {e}")
        
        if filter_tools:
            logger.info(f"=== TOOLS FILTERED: {filtered_count} filtered out, {len(all_tools)} allowed ===")
        logger.info(f"=== TOTAL FASTMCP TOOLS AVAILABLE TO LLM: {len(all_tools)} ===")
        return all_tools
    
    async def call_tool(self, tool_name: str, arguments: Dict = None, server_id: str = None) -> Dict:
        """Call a tool using FastMCP"""
        if arguments is None:
            arguments = {}
            
        await self.initialize()
        
        if not self.client:
            return {"error": "FastMCP client not initialized"}
        
        try:
            logger.info(f"=== FASTMCP TOOL CALL ===")
            logger.info(f"Tool: {tool_name}")
            logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")
            
            async with self.client as client:
                result = await client.call_tool(tool_name, arguments)
                
                logger.info(f"FastMCP tool call successful")
                logger.info(f"Raw result type: {type(result)}")
                logger.info(f"Raw result: {result}")
                
                # Handle different FastMCP result formats
                if hasattr(result, 'content') and result.content:
                    # Result has content attribute (list of TextContent objects)
                    if isinstance(result.content, list):
                        # Extract text from TextContent objects
                        text_parts = []
                        for content_item in result.content:
                            if hasattr(content_item, 'text'):
                                text_parts.append(content_item.text)
                            elif isinstance(content_item, str):
                                text_parts.append(content_item)
                            else:
                                text_parts.append(str(content_item))
                        tool_result = "\n".join(text_parts)
                        logger.info(f"Extracted text from content list: {tool_result}")
                        return {"result": tool_result}
                    else:
                        # Single content item
                        if hasattr(result.content, 'text'):
                            tool_result = result.content.text
                        else:
                            tool_result = str(result.content)
                        logger.info(f"Extracted content: {tool_result}")
                        return {"result": tool_result}
                elif hasattr(result, 'data'):
                    # Result has data attribute
                    tool_result = result.data
                    logger.info(f"Extracted data: {tool_result}")
                    return {"result": tool_result}
                elif isinstance(result, dict):
                    # Result is already a dictionary
                    logger.info(f"Result is dict: {result}")
                    return {"result": result}
                elif isinstance(result, list):
                    # Result is a list (multiple content items)
                    logger.info(f"Result is list: {result}")
                    if len(result) > 0 and hasattr(result[0], 'text'):
                        # Extract text from content items
                        tool_result = "\n".join([item.text for item in result if hasattr(item, 'text')])
                        logger.info(f"Extracted text from list: {tool_result}")
                        return {"result": tool_result}
                    else:
                        return {"result": result}
                else:
                    # Fallback - convert to string
                    tool_result = str(result)
                    logger.info(f"Converted to string: {tool_result}")
                    return {"result": tool_result}
                    
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup FastMCP client"""
        logger.info("Cleaning up FastMCP client...")
        # FastMCP handles cleanup automatically with async context managers
        self.client = None
        self._initialized = False

class WorkshopRAGChatbot:
    """Main chatbot with embedded RAG and MCP integration"""

    def __init__(self):
        self.rag_engine = SimpleRAGEngine()
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self._initialized = False
        self.system_prompt_config = self._load_system_prompt_config()
        self.mcp_config = self._load_mcp_config()
        self.mcp_manager = FastMCPManager(self.mcp_config)

        logger.info("Workshop RAG chatbot initialized with FastMCP")

    def _load_system_prompt_config(self) -> Dict:
        """Load system prompt configuration from environment variable"""
        try:
            config_yaml = os.getenv("SYSTEM_PROMPT_CONFIG")
            if config_yaml:
                config = yaml.safe_load(config_yaml)
                logger.info("Loaded system prompt configuration from environment variable")
                return config
            else:
                logger.warning("SYSTEM_PROMPT_CONFIG environment variable not found")
                return self._get_default_system_prompt_config()
        except Exception as e:
            logger.error(f"Error loading system prompt config from environment: {e}")
            return self._get_default_system_prompt_config()

    def _get_default_system_prompt_config(self) -> Dict:
        """Fallback system prompt configuration if external file is not available"""
        return {
            "workshop": {
                "title": "Workshop",
                "focus": "Technical Training"
            },
            "system_prompt": {
                "introduction": "You are a helpful AI assistant for the workshop.",
                "special_instructions": "",
                "guidelines": [
                    "Be helpful, concise, and technically accurate",
                    "Reference workshop content when relevant",
                    "Provide step-by-step guidance"
                ],
                "response_format": {
                    "description": "Format responses clearly and professionally",
                    "rules": [
                        "Use clear structure",
                        "Include examples when helpful"
                    ]
                },
                "mcp_instructions": "Use available tools when appropriate"
            }
        }

    def _load_mcp_config(self) -> Dict:
        """Load MCP servers configuration from environment variable"""
        try:
            config_yaml = os.getenv("MCP_CONFIG")
            if config_yaml:
                config = yaml.safe_load(config_yaml)
                logger.info("Loaded MCP configuration from environment variable")
                return config
            else:
                logger.warning("MCP_CONFIG environment variable not found")
                return self._get_default_mcp_config()
        except Exception as e:
            logger.error(f"Error loading MCP config from environment: {e}")
            return self._get_default_mcp_config()

    def _get_default_mcp_config(self) -> Dict:
        """Fallback MCP configuration if external file is not available"""
        return {
            "mcp_servers": {}
        }

    def get_enabled_mcp_servers(self) -> List[Dict]:
        """Get list of all configured MCP servers (FastMCP format)"""
        servers = []
        
        mcp_servers = self.mcp_config.get("mcpServers", {})
        for server_id, server_config in mcp_servers.items():
            servers.append({
                "id": server_id,
                "name": server_config.get("name", server_id),
                "url": server_config.get("url", ""),
                "description": server_config.get("description", ""),
                "priority": server_config.get("priority", 99),
                "transport": server_config.get("transport", "http")
            })

        # Sort by priority
        servers.sort(key=lambda x: x["priority"])
        return servers


    async def _initialize_rag(self):
        """Initialize RAG engine with workshop content"""
        if self._initialized:
            return

        try:
            logger.info("Initializing RAG engine with workshop content...")
            documents = await self._extract_workshop_content()
            self.rag_engine.ingest_documents(documents)
            self._initialized = True
            logger.info("RAG engine initialization completed")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")

    async def _ensure_initialized(self):
        """Ensure RAG is initialized before use"""
        if not self._initialized:
            await self._initialize_rag()

    async def _extract_workshop_content(self) -> List[Dict]:
        """Extract content from workshop AsciiDoc source files"""
        documents = []
        content_path = Path(os.getenv("CONTENT_DIR", "/app/content"))

        if not content_path.exists():
            logger.warning(f"Content directory not found: {content_path}")
            # No fallback to HTML - only use AsciiDoc and PDF content
            logger.info(f"Extracted 0 documents from AsciiDoc workshop content (no content directory)")
            return await self._extract_pdf_content()

        # Find all AsciiDoc files in modules
        modules_dir = content_path / "modules" / "ROOT" / "pages"
        if modules_dir.exists():
            adoc_files = list(modules_dir.glob("**/*.adoc"))

            logger.info(f"Found {len(adoc_files)} AsciiDoc files to process")
            for adoc_file in adoc_files:
                try:
                    logger.debug(f"Reading AsciiDoc file: {adoc_file}")
                    content = adoc_file.read_text(encoding='utf-8')

                    # Extract title from AsciiDoc (first line starting with = or explicit title)
                    title = self._extract_adoc_title(content, adoc_file.stem)

                    # Clean up AsciiDoc markup for better RAG processing
                    cleaned_content = self._clean_asciidoc_content(content)

                    if len(cleaned_content.strip()) > 100:
                        # Determine module from path
                        module = self._extract_module_from_path(str(adoc_file))

                        documents.append({
                            'title': title,
                            'content': cleaned_content,
                            'module': module,
                            'file_path': str(adoc_file),
                            'content_type': self._determine_adoc_content_type(str(adoc_file))
                        })
                        logger.debug(f"  ✓ Added: {title} ({len(cleaned_content)} chars)")
                    else:
                        logger.debug(f"  ✗ Skipped {adoc_file}: content too short ({len(cleaned_content.strip())} chars)")

                except Exception as e:
                    logger.warning(f"Error processing {adoc_file}: {e}")
                    continue

        # Also extract PDF documentation files
        pdf_documents = await self._extract_pdf_content()
        documents.extend(pdf_documents)

        logger.info(f"Extracted {len(documents)} documents from AsciiDoc workshop content and PDF files")
        return documents


    async def _extract_pdf_content(self) -> List[Dict]:
        """Extract content from PDF documentation files"""
        documents = []
        pdf_path = Path(os.getenv("PDF_DIR", "/app/pdfs"))

        if not pdf_path.exists():
            logger.warning(f"PDF directory not found: {pdf_path}")
            return documents

        # Find all PDF files
        pdf_files = list(pdf_path.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing PDF: {pdf_file}")

                # Extract text from PDF
                reader = PdfReader(str(pdf_file))
                pdf_text = ""
                pages_processed = 0

                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                            pages_processed += 1
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1} from {pdf_file}: {e}")
                        continue
                
                logger.info(f"  Extracted text from {pages_processed} pages")

                if len(pdf_text.strip()) > 100:
                    # Extract title from filename
                    title = self._extract_pdf_title(pdf_file.name)

                    # Clean the extracted text
                    cleaned_content = self._clean_pdf_content(pdf_text)

                    # Split large PDFs into chunks for better RAG performance
                    chunks = self._chunk_pdf_content(cleaned_content, title)

                    logger.info(f"  Split into {len(chunks)} chunks")
                    for i, chunk in enumerate(chunks):
                        if len(chunk['content'].strip()) > 100:
                            chunk_title = f"{title}"
                            if len(chunks) > 1:
                                chunk_title += f" (Part {i+1})"

                            documents.append({
                                'title': chunk_title,
                                'content': chunk['content'],
                                'module': 'PDF Documentation',
                                'file_path': str(pdf_file),
                                'content_type': 'pdf-documentation',
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'processing_method': 'pdf'
                            })

            except Exception as e:
                logger.warning(f"Error processing PDF {pdf_file}: {e}")
                continue

        logger.info(f"Extracted {len(documents)} documents from PDF files")
        return documents

    def _extract_pdf_title(self, filename: str) -> str:
        """Extract title from PDF filename"""
        # Remove extension and clean up filename
        title = filename.replace('.pdf', '')
        title = title.replace('_', ' ').replace('-', ' ')

        # Handle Red Hat Developer Hub documentation naming pattern
        if 'Red_Hat_Developer_Hub' in filename:
            parts = filename.split('-')
            if len(parts) >= 3:
                doc_title = parts[2].replace('_', ' ')
                return f"Red Hat Developer Hub - {doc_title}"

        return title.title()

    def _clean_pdf_content(self, content: str) -> str:
        """Clean PDF extracted content"""
        if not content:
            return ""

        # Remove page markers
        content = re.sub(r'\n--- Page \d+ ---\n', '\n', content)

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)

        # Remove common PDF artifacts
        content = re.sub(r'\f', '\n', content)  # Form feed characters
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', content)  # Control characters

        return content.strip()

    def _chunk_pdf_content(self, content: str, title: str) -> List[Dict]:
        """Split PDF content into manageable chunks"""
        chunks = []

        # Split by double newlines to get paragraphs/sections
        sections = content.split('\n\n')

        current_chunk = ""
        chunk_size_limit = 2000  # Characters per chunk

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If adding this section would exceed the limit, save current chunk
            if len(current_chunk) + len(section) > chunk_size_limit and current_chunk:
                chunks.append({'content': current_chunk.strip()})
                current_chunk = section
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({'content': current_chunk.strip()})

        # If no chunks were created, create one with the full content (truncated if necessary)
        if not chunks:
            content_truncated = content[:chunk_size_limit] if len(content) > chunk_size_limit else content
            chunks.append({'content': content_truncated})

        return chunks

    def _extract_adoc_title(self, content: str, fallback: str) -> str:
        """Extract title from AsciiDoc content"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # Look for main title (= Title) or section title (== Title)
            if line.startswith('= ') and not line.startswith('== '):
                return line[2:].strip()
            elif line.startswith('== '):
                return line[3:].strip()
        return fallback.replace('-', ' ').title()

    def _clean_asciidoc_content(self, content: str) -> str:
        """Clean AsciiDoc markup for better RAG processing"""
        # Remove AsciiDoc attributes and metadata at the top
        lines = content.split('\n')
        cleaned_lines = []
        in_header = True

        for line in lines:
            stripped = line.strip()

            # Skip metadata/attributes at the start
            if in_header and (
                stripped.startswith(':') or
                stripped.startswith('//') or
                stripped == '' or
                stripped.startswith('= ')
            ):
                if stripped.startswith('= '):
                    in_header = False
                    cleaned_lines.append(stripped[2:])  # Add title without markup
                continue
            else:
                in_header = False

            # Clean common AsciiDoc markup
            cleaned = line

            # Remove section headers markup but keep the text
            cleaned = re.sub(r'^=+\s+', '', cleaned)

            # Remove inline markup
            cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)  # Bold
            cleaned = re.sub(r'_([^_]+)_', r'\1', cleaned)    # Italic
            cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)    # Code

            # Remove links markup but keep text
            cleaned = re.sub(r'link:([^\[]+)\[([^\]]*)\]', r'\2', cleaned)
            cleaned = re.sub(r'https?://[^\s\[\]]+\[([^\]]*)\]', r'\1', cleaned)

            # Remove image references
            cleaned = re.sub(r'image::?[^\[]+\[[^\]]*\]', '', cleaned)

            # Remove include directives
            cleaned = re.sub(r'include::[^\[]+\[[^\]]*\]', '', cleaned)

            # Remove block delimiters
            cleaned = re.sub(r'^[-=\*]{3,}$', '', cleaned)

            # Remove source block markers
            cleaned = re.sub(r'^\[source[^\]]*\]$', '', cleaned)

            # Remove empty attribute lines
            cleaned = re.sub(r'^\[[^\]]*\]$', '', cleaned)

            # Clean up extra whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            if cleaned:
                cleaned_lines.append(cleaned)

        return '\n'.join(cleaned_lines)

    def _extract_module_from_path(self, file_path: str) -> str:
        """Extract module name from file path"""
        match = re.search(r'/m(\d+)/', file_path)
        if match:
            return f"Module {match.group(1)}"
        return "General"


    def _determine_adoc_content_type(self, file_path: str) -> str:
        """Determine content type from AsciiDoc file path"""
        filename = Path(file_path).name
        if filename == "index.adoc":
            return "index"
        elif re.search(r'module-\d+\.0\.adoc', filename):
            return "module-overview"
        elif re.search(r'module-\d+\.\d+\.adoc', filename):
            return "module-section"
        elif filename.startswith("module-"):
            return "module-content"
        return "page"

    async def retrieve_relevant_content(self, query: str, page_context: str = None, max_chunks: int = 3) -> str:
        """Retrieve relevant content using RAG with optional page context prioritization"""
        await self._ensure_initialized()
        
        # Enhance query with page context if provided
        enhanced_query = query
        if page_context:
            enhanced_query = f"{page_context} {query}"
            logger.info(f"RAG ENHANCED QUERY: Adding page context '{page_context}' to query '{query}'")
        
        results = self.rag_engine.search(enhanced_query, top_k=max_chunks * 2)  # Get more results for filtering

        if not results:
            logger.info(f"RAG QUERY: '{enhanced_query}' - No relevant content found, using fallback")
            return (
                "Platform Engineering Workshop covering Red Hat Developer Hub (RHDH), Backstage, "
                "Internal Developer Platforms (IDP), software templates, catalog management, "
                "GitOps with ArgoCD, Kubernetes integration, authentication with Keycloak, "
                "and developer self-service capabilities."
            )

        # Prioritize results based on page context
        if page_context:
            prioritized_results = self._prioritize_results_by_page_context(results, page_context, max_chunks)
        else:
            prioritized_results = results[:max_chunks]

        # Log detailed RAG debugging information
        logger.info(f"=== RAG SEARCH START ===")
        logger.info(f"RAG QUERY: '{query}' with page context: '{page_context}'")
        logger.info(f"RAG RESULTS: Found {len(prioritized_results)} relevant chunks:")
        logger.info(f"=== RAG CONTENT SNIPPETS ===")

        # Combine relevant content
        relevant_content = []
        for i, result in enumerate(prioritized_results, 1):
            metadata = result['metadata']
            similarity = result['similarity']
            content = result['content']  # Don't truncate for debug logging

            # Log each retrieved chunk with full content
            logger.info(f"[{i}] {metadata['title']} (similarity: {similarity:.3f})")
            logger.info(f"    File: {metadata['file_path']}")
            logger.info(f"    Module: {metadata['module']}")
            logger.info(f"    Full content ({len(content)} chars):")
            logger.info(f"    --- SNIPPET START ---")
            logger.info(content)
            logger.info(f"    --- SNIPPET END ---")

            # Truncate for context window when adding to relevant_content
            truncated_content = content[:500] if len(content) > 500 else content
            relevant_content.append(
                f"[{metadata['module']} - {metadata['title']}]\n{truncated_content}"
            )

        context = "\n\n---\n\n".join(relevant_content)
        logger.info(f"=== RAG SEARCH COMPLETE ===")
        logger.info(f"RAG CONTEXT: Combined context length: {len(context)} characters")
        logger.info(f"RAG CONTEXT: Using {len(relevant_content)} content snippets")

        return context
    
    def _prioritize_results_by_page_context(self, results: List[Dict], page_context: str, max_chunks: int) -> List[Dict]:
        """Prioritize RAG results based on page context"""
        if not page_context:
            return results[:max_chunks]
        
        page_context_lower = page_context.lower()
        prioritized = []
        remaining = []
        
        for result in results:
            metadata = result['metadata']
            title_lower = metadata.get('title', '').lower()
            module_lower = metadata.get('module', '').lower()
            
            # Check if this chunk is related to the current page
            is_related = (
                page_context_lower in title_lower or
                any(word in title_lower for word in page_context_lower.split() if len(word) > 3) or
                page_context_lower in module_lower
            )
            
            if is_related:
                prioritized.append(result)
                logger.debug(f"PRIORITIZED: {metadata['title']} (matches page context)")
            else:
                remaining.append(result)
        
        # Take prioritized results first, then fill with remaining
        final_results = prioritized[:max_chunks]
        if len(final_results) < max_chunks:
            final_results.extend(remaining[:max_chunks - len(final_results)])
        
        logger.info(f"RAG PRIORITIZATION: {len(prioritized)} page-related, {len(remaining)} general, returning {len(final_results)} total")
        return final_results

    async def get_mcp_tools(self, server_url: str = None) -> List[Dict]:
        """Get available MCP tools using FastMCP (simplified interface)"""
        # FastMCP doesn't support single-server queries in our current setup
        # Always return all tools from all configured servers
        return await self.mcp_manager.get_all_tools()


    async def call_mcp_tool(self, tool_name: str, parameters: Dict = None, server_url: str = None) -> Dict:
        """Call an MCP tool using FastMCP (simplified interface)"""
        return await self.mcp_manager.call_tool(tool_name, parameters)


    async def stream_chat_response(self, user_message: str, conversation_history: List[ConversationMessage] = None, include_mcp: bool = True, page_context: str = None) -> AsyncGenerator[str, None]:
        """Generate streaming chat response with RAG and optional page context"""
        try:
            logger.info("=== CHAT REQUEST START ===")
            logger.info(f"User Message: {user_message}")
            logger.info(f"Include MCP: {include_mcp}")
            logger.info(f"Page Context: {page_context}")
            logger.info(f"Conversation History Length: {len(conversation_history) if conversation_history else 0}")

            # Step 1: Retrieve relevant context using RAG with page context
            if page_context:
                yield f"data: {json.dumps({'status': f'Searching {page_context} content...'})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'Searching workshop content...'})}\n\n"
            relevant_context = await self.retrieve_relevant_content(user_message, page_context)

            # Step 2: Build system prompt with retrieved context
            system_prompt = self._build_system_prompt(relevant_context, include_mcp, page_context)
            logger.info(f"=== SYSTEM PROMPT ===")
            logger.info(f"System Prompt: {system_prompt}")
            logger.info("=== END SYSTEM PROMPT ===")

            # Step 3: Prepare messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history
            if conversation_history:
                for msg in conversation_history:
                    messages.append({"role": msg.role, "content": msg.content})

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            logger.info(f"=== COMPLETE MESSAGE CHAIN ===")
            logger.info(f"Total Messages: {len(messages)}")
            for i, msg in enumerate(messages):
                logger.info(f"Message {i+1}: Role={msg['role']}, Content={msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
            logger.info("=== END MESSAGE CHAIN ===")

            # Step 4: Generate response with tools if MCP is enabled
            if include_mcp:
                logger.info("=== USING TOOLS PATH ===")
                yield f"data: {json.dumps({'status': 'Generating response with tools...'})}\n\n"
                async for chunk in self._stream_with_tools(messages, user_message):
                    yield chunk
            else:
                logger.info("=== USING REGULAR STREAMING PATH ===")
                yield f"data: {json.dumps({'status': 'Generating response...'})}\n\n"
                async for chunk in self._stream_llm_response(messages):
                    yield chunk

            logger.info("=== CHAT REQUEST COMPLETED ===")

        except Exception as e:
            logger.error(f"Error in stream_chat_response: {e}")
            logger.info("=== CHAT REQUEST FAILED ===")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def _build_system_prompt(self, relevant_context: str, include_mcp: bool, page_context: str = None) -> str:
        """Build system prompt with RAG context, tool info, and optional page context using external configuration"""
        config = self.system_prompt_config
        workshop_title = config.get("workshop", {}).get("title", "Workshop")

        # Build the system prompt from configuration
        prompt_config = config.get("system_prompt", {})

        # Introduction section
        introduction = prompt_config.get("introduction", "You are a helpful AI assistant.")
        introduction = introduction.format(workshop_title=workshop_title)

        prompt = f"{introduction}\n\n"
        
        # Add page context information if provided
        if page_context:
            prompt += f"CURRENT PAGE CONTEXT:\nThe user is currently viewing: {page_context}\n"
            prompt += "Focus your response on content and guidance relevant to this specific page/topic.\n\n"
        
        prompt += f"RELEVANT WORKSHOP CONTEXT:\n{relevant_context}\n\n"

        # Special instructions (workshop-specific)
        special_instructions = prompt_config.get("special_instructions", "")
        if special_instructions:
            prompt += f"{special_instructions}\n\n"

        # Guidelines
        guidelines = prompt_config.get("guidelines", [])
        if guidelines:
            prompt += "GUIDELINES:\n"
            for guideline in guidelines:
                prompt += f"- {guideline}\n"
            prompt += "\n"

        # Response format
        response_format = prompt_config.get("response_format", {})
        format_description = response_format.get("description", "")
        format_rules = response_format.get("rules", [])
        format_example = response_format.get("example", "")

        if format_description or format_rules:
            prompt += "RESPONSE FORMAT:\n"
            if format_description:
                prompt += f"{format_description}\n"

            for rule in format_rules:
                prompt += f"- {rule}\n"

            if format_example:
                prompt += f"\nExample format:\n{format_example}\n"

            prompt += "\n"

        # MCP instructions
        if include_mcp:
            mcp_instructions = prompt_config.get("mcp_instructions", "")
            if mcp_instructions:
                prompt += f"{mcp_instructions}\n"

        return prompt

    async def _stream_llm_response(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """Stream response from LLM"""
        try:
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "max_completion_tokens": config.MAX_TOKENS,
                "stream": True
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.LLM_API_KEY[:10]}...{config.LLM_API_KEY[-4:]}"  # Partial key for debugging
            }

            # Log the complete request
            logger.info("=== LLM REQUEST ===")
            logger.info(f"URL: {config.LLM_API_URL}")
            logger.info(f"Headers: {json.dumps(headers, indent=2)}")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            logger.info("=== END REQUEST ===")

            # Set proper auth header for actual request
            headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"

            async with self.http_client.stream("POST", config.LLM_API_URL, json=payload, headers=headers) as response:
                logger.info(f"=== LLM RESPONSE STATUS ===")
                logger.info(f"Status Code: {response.status_code}")
                logger.info(f"Response Headers: {dict(response.headers)}")

                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"LLM API Error Response: {error_text.decode()}")
                    yield f"data: {json.dumps({'error': f'LLM API error: {response.status_code}'})}\n\n"
                    return

                logger.info("=== LLM STREAMING RESPONSE ===")
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]
                        logger.debug(f"Raw SSE line: {line}")

                        if data_str.strip() == "[DONE]":
                            logger.info("LLM stream completed with [DONE]")
                            break

                        try:
                            data = json.loads(data_str)
                            logger.debug(f"Parsed SSE data: {json.dumps(data, indent=2)}")

                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse SSE data: {data_str}, error: {e}")
                            continue
                logger.info("=== END LLM RESPONSE ===")

        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def _stream_with_tools(self, messages: List[Dict], user_message: str) -> AsyncGenerator[str, None]:
        """Stream response with MCP tool integration"""
        try:
            # Get available tools
            tools = await self.get_mcp_tools()
            if not tools:
                # Fall back to regular streaming
                async for chunk in self._stream_llm_response(messages):
                    yield chunk
                return

            # Format tools for function calling
            formatted_tools = []
            for tool in tools:
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", f"Execute {tool['name']} operation"),
                        "parameters": tool.get("inputSchema", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                }
                formatted_tools.append(formatted_tool)

            # Call LLM with tools
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "max_completion_tokens": config.MAX_TOKENS,
                "tools": formatted_tools,
                "tool_choice": "auto"
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.LLM_API_KEY[:10]}...{config.LLM_API_KEY[-4:]}"  # Partial key for debugging
            }

            # Log the complete tool request
            logger.info("=== LLM TOOL REQUEST ===")
            logger.info(f"URL: {config.LLM_API_URL}")
            logger.info(f"Headers: {json.dumps(headers, indent=2)}")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            logger.info("=== END TOOL REQUEST ===")

            # Set proper auth header for actual request
            headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"

            response = await self.http_client.post(config.LLM_API_URL, json=payload, headers=headers)

            logger.info(f"=== LLM TOOL RESPONSE STATUS ===")
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Response Headers: {dict(response.headers)}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"=== LLM TOOL RESPONSE DATA ===")
                logger.info(f"Response Data: {json.dumps(data, indent=2)}")
                logger.info("=== END TOOL RESPONSE DATA ===")

                choice = data["choices"][0]
                message = choice["message"]

                # Check if tools were called
                if "tool_calls" in message and message["tool_calls"]:
                    logger.info(f"=== TOOL CALLS DETECTED ===")
                    logger.info(f"Number of tool calls: {len(message['tool_calls'])}")

                    # Execute tool calls
                    for i, tool_call in enumerate(message["tool_calls"]):
                        tool_name = tool_call["function"]["name"]
                        args = json.loads(tool_call["function"]["arguments"])

                        logger.info(f"=== EXECUTING TOOL CALL {i+1} ===")
                        logger.info(f"Tool Name: {tool_name}")
                        logger.info(f"Tool Arguments: {json.dumps(args, indent=2)}")
                        logger.info(f"Tool Call ID: {tool_call['id']}")

                        yield f"data: {json.dumps({'status': f'Executing {tool_name}...'})}\n\n"

                        result = await self.call_mcp_tool(tool_name, args)

                        logger.info(f"=== TOOL RESULT {i+1} ===")
                        logger.info(f"Tool Result: {json.dumps(result, indent=2)}")
                        logger.info("=== END TOOL RESULT ===")

                        # Add tool result to conversation
                        messages.append({
                            "role": "assistant",
                            "tool_calls": [tool_call]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(result)
                        })

                    logger.info("=== GENERATING FINAL RESPONSE WITH TOOL RESULTS ===")
                    # Generate final response with tool results
                    async for chunk in self._stream_llm_response(messages):
                        yield chunk
                else:
                    logger.info("=== NO TOOL CALLS DETECTED ===")
                    # No tools called, stream the response
                    content = message.get("content", "")
                    logger.info(f"Direct response content: {content}")
                    yield f"data: {json.dumps({'content': content})}\n\n"
            else:
                # Fall back to regular streaming
                async for chunk in self._stream_llm_response(messages):
                    yield chunk

        except Exception as e:
            logger.error(f"Error in tool integration: {e}")
            yield f"data: {json.dumps({'error': f'Tool integration error: {str(e)}'})}\n\n"

# Initialize chatbot
chatbot = WorkshopRAGChatbot()

# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Application starting up...")
    # MCP servers will be initialized as needed
    
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown"""
    logger.info("Application shutting down...")
    if chatbot.http_client:
        await chatbot.http_client.aclose()
    # Cleanup FastMCP clients
    chatbot.mcp_manager.cleanup()

# API Routes
@app.post("/api/chat/stream")
async def stream_chat(chat_request: ChatRequest):
    """Stream chat responses using Server-Sent Events with embedded RAG"""

    async def generate():
        yield "data: {\"status\": \"starting\"}\n\n"

        async for chunk in chatbot.stream_chat_response(
            chat_request.message,
            chat_request.conversation_history,
            chat_request.include_mcp,
            chat_request.page_context
        ):
            yield chunk

        yield "data: {\"status\": \"complete\"}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.get("/api/mcp/tools")
async def get_mcp_tools():
    """Get available MCP tools"""
    tools = await chatbot.get_mcp_tools()
    return {"tools": tools}

@app.get("/api/rag/search")
async def search_content(q: str, limit: int = 5, debug: bool = False):
    """Search workshop content using embedded RAG"""
    await chatbot._ensure_initialized()
    results = chatbot.rag_engine.search(q, top_k=limit)

    response = {
        "query": q,
        "results": [
            {
                "title": r["metadata"]["title"],
                "module": r["metadata"]["module"],
                "content_type": r["metadata"]["content_type"],
                "similarity": r["similarity"],
                "content_preview": r["content"][:200] + "...",
                "content_length": len(r["content"])
            } for r in results
        ],
        "count": len(results)
    }

    if debug:
        response["debug"] = {
            "total_documents": len(chatbot.rag_engine.documents),
            "rag_initialized": chatbot.rag_engine.is_initialized,
            "cleaned_query": chatbot.rag_engine._clean_text(q) if results else None,
            "full_results": [
                {
                    "title": r["metadata"]["title"],
                    "module": r["metadata"]["module"],
                    "similarity": r["similarity"],
                    "file_path": r["metadata"]["file_path"],
                    "full_content": r["content"]
                } for r in results
            ] if results else []
        }

    return response

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    rag_status = "initialized" if chatbot._initialized and chatbot.rag_engine.is_initialized else "not_initialized"
    doc_count = len(chatbot.rag_engine.documents) if chatbot.rag_engine.is_initialized else 0

    # Get MCP server info
    mcp_servers = chatbot.get_enabled_mcp_servers()
    mcp_info = {
        "configured_servers": len(mcp_servers),
        "servers": [{"name": s["name"], "url": s["url"]} for s in mcp_servers]
    }

    return {
        "status": "healthy",
        "version": "2.0.0-embedded-rag",
        "config": {
            "llm_model": config.LLM_MODEL,
            "static_dir": config.STATIC_DIR,
        },
        "mcp": mcp_info,
        "rag": {
            "status": rag_status,
            "document_count": doc_count,
            "engine": "tfidf-cosine"
        }
    }

# Serve static files
@app.get("/")
async def serve_index():
    """Serve the main index page"""
    static_path = Path(config.STATIC_DIR) / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    raise HTTPException(status_code=404, detail="Index page not found")

# Mount static files for all other paths
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)