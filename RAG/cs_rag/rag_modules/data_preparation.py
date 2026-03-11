import os
import json
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from loguru import logger
from ..config import Config


class DataPreparationModule:
    """数据准备模块 - 负责文档加载、清洗、分块和元数据提取"""

    def __init__(self):
        self.knowledge_base_dir = Config.KNOWLEDGE_BASE_DIR
        self.metadata_dir = Config.METADATA_DIR
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        
        logger.info("数据准备模块初始化完成")

    def load_documents(self) -> List[Any]:
        """加载知识库中的所有文档"""
        documents = []
        
        for file_path in self.knowledge_base_dir.glob("**/*"):
            if file_path.suffix.lower() in ['.pdf', '.txt', '.md']:
                try:
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                        docs = loader.load()
                    elif file_path.suffix.lower() == '.txt':
                        loader = TextLoader(str(file_path), encoding='utf-8')
                        docs = loader.load()
                    elif file_path.suffix.lower() == '.md':
                        loader = UnstructuredMarkdownLoader(str(file_path))
                        docs = loader.load()
                    
                    # 添加来源信息到元数据
                    for doc in docs:
                        doc.metadata['source'] = str(file_path)
                        
                    documents.extend(docs)
                    logger.info(f"成功加载文档: {file_path}")
                    
                except Exception as e:
                    logger.error(f"加载文档失败 {file_path}: {str(e)}")
        
        return documents

    def clean_text(self, text: str) -> str:
        """清洗文本，移除多余空白、特殊字符等"""
        # 移除多余的空白字符
        cleaned_text = ' '.join(text.split())
        return cleaned_text

    def split_markdown_documents(self, documents: List[Any]) -> List[Any]:
        """使用Markdown标题分割器对文档进行结构化分割"""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        
        split_docs = []
        chunk_id = 0
        
        for doc in documents:
            if doc.metadata.get('source', '').endswith('.md'):
                # 对Markdown文档使用特殊分割器
                split_result = splitter.split_text(doc.page_content)
                
                for i, chunk in enumerate(split_result):
                    # 为每个块生成唯一ID
                    chunk_id += 1
                    
                    # 更新元数据，添加块ID和标题信息
                    chunk.metadata.update({
                        'chunk_id': chunk_id,
                        'source_document': doc.metadata.get('source'),
                        'original_page': doc.metadata.get('page', 1),
                        'headers': chunk.metadata.get('headers', {})
                    })
                    
                    split_docs.append(chunk)
            else:
                # 对非Markdown文档使用常规分割
                split_docs.append(doc)
        
        logger.info(f"文档分割完成，共生成 {len(split_docs)} 个块")
        return split_docs

    def extract_metadata(self, documents: List[Any]) -> Dict[str, Any]:
        """提取文档元数据并导出到JSON文件"""
        metadata_list = []
        
        for idx, doc in enumerate(documents):
            metadata = {
                'chunk_id': getattr(doc, 'metadata', {}).get('chunk_id', idx),
                'source_file': doc.metadata.get('source_document') or doc.metadata.get('source'),
                'original_page': doc.metadata.get('original_page', 1),
                'headers': doc.metadata.get('headers', {}),
                'length': len(doc.page_content),
                'created_at': str(Path(doc.metadata.get('source_document') or doc.metadata.get('source', '')).stat().st_ctime)
            }
            
            metadata_list.append(metadata)
        
        # 导出到JSON文件
        metadata_file_path = self.metadata_dir / "document_metadata.json"
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {metadata_file_path}, 共 {len(metadata_list)} 条记录")
        return metadata_list

    def filter_documents(self, documents: List[Any], filters: Dict[str, Any] = None) -> List[Any]:
        """根据分类等对文档进行过滤"""
        if not filters:
            return documents
            
        filtered_docs = []
        for doc in documents:
            # 这里可以根据filters中的条件进行过滤
            # 例如：按类别、标签、日期等过滤
            include_doc = True
            
            # 示例过滤逻辑（可根据实际需求修改）
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            include_doc = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            include_doc = False
                            break
            
            if include_doc:
                filtered_docs.append(doc)
        
        logger.info(f"文档过滤完成，剩余 {len(filtered_docs)} 个文档")
        return filtered_docs

    def validate_chunks(self, documents: List[Any]) -> List[Any]:
        """检查分块质量，过滤空块或过短块"""
        validated_docs = []
        
        for doc in documents:
            # 过滤空块或过短块（少于20个字符）
            if len(doc.page_content.strip()) > 20:
                validated_docs.append(doc)
            else:
                logger.warning(f"跳过过短的文档块: {doc.metadata.get('chunk_id', 'unknown')}")
        
        logger.info(f"块验证完成，保留 {len(validated_docs)} 个有效块")
        return validated_docs

    def prepare_data(self, filters: Dict[str, Any] = None) -> tuple[List[Any], Dict[str, Any]]:
        """执行完整的数据准备流程"""
        logger.info("开始数据准备流程")
        
        # 1. 加载文档
        raw_docs = self.load_documents()
        
        # 2. 清洗文本
        cleaned_docs = [self.clean_text(doc.page_content) for doc in raw_docs]
        
        # 重新创建Document对象
        from langchain.docstore.document import Document
        cleaned_doc_objects = []
        for i, doc in enumerate(raw_docs):
            cleaned_doc = Document(
                page_content=cleaned_docs[i],
                metadata=doc.metadata
            )
            cleaned_doc_objects.append(cleaned_doc)
        
        # 3. 对Markdown文档进行结构化分割
        split_docs = self.split_markdown_documents(cleaned_doc_objects)
        
        # 4. 应用过滤器
        if filters:
            split_docs = self.filter_documents(split_docs, filters)
        
        # 5. 验证块质量
        final_docs = self.validate_chunks(split_docs)
        
        # 6. 提取元数据
        metadata = self.extract_metadata(final_docs)
        
        logger.info(f"数据准备完成，共 {len(final_docs)} 个有效文档块")
        return final_docs, metadata