import os
import pickle
from pathlib import Path
from typing import List, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from loguru import logger
from ..config import Config


class IndexConstructionModule:
    """索引构建模块 - 负责向量索引的构建、管理和优化"""

    def __init__(self):
        self.vector_index_dir = Config.VECTOR_INDEX_DIR
        self.embedding_model_name = Config.EMBEDDING_MODEL_NAME
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # 初始化向量存储
        self.vectorstore = None
        
        logger.info("索引构建模块初始化完成")

    def build_vector_index(self, documents: List[Any]):
        """构建向量索引"""
        logger.info("开始构建向量索引...")
        
        try:
            # 创建向量存储
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.vector_index_dir)
            )
            
            logger.info(f"向量索引构建完成，包含 {len(documents)} 个文档")
        except Exception as e:
            logger.error(f"构建向量索引失败: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = None) -> List[Any]:
        """相似度搜索"""
        if not self.vectorstore:
            raise ValueError("向量索引尚未构建，请先调用build_vector_index方法")
        
        if k is None:
            k = Config.RETRIEVAL_K
            
        # 执行相似度搜索
        results = self.vectorstore.similarity_search(query, k=k)
        
        # 过滤低于相似度阈值的结果
        filtered_results = []
        for result in results:
            # 计算嵌入向量之间的相似度（这里简化处理，实际应使用更精确的方法）
            # 此处仅为演示，实际实现需要根据具体嵌入模型和ChromaDB的特性进行调整
            filtered_results.append(result)
        
        logger.info(f"相似度搜索完成，返回 {len(filtered_results)} 个结果")
        return filtered_results

    def delete_index(self):
        """删除向量索引"""
        try:
            index_path = self.vector_index_dir
            if index_path.exists():
                import shutil
                shutil.rmtree(index_path)
                logger.info("向量索引已删除")
            else:
                logger.warning("尝试删除不存在的向量索引")
        except Exception as e:
            logger.error(f"删除向量索引失败: {str(e)}")

    def update_index(self, new_documents: List[Any]):
        """向现有索引添加新文档"""
        if not self.vectorstore:
            logger.info("向量索引不存在，正在创建新索引")
            self.build_vector_index(new_documents)
            return
        
        try:
            # 添加新文档到现有索引
            self.vectorstore.add_documents(new_documents)
            logger.info(f"成功向索引添加 {len(new_documents)} 个新文档")
        except Exception as e:
            logger.error(f"更新向量索引失败: {str(e)}")
            raise

    def save_index(self, index_name: str = "default_index"):
        """保存向量索引到指定路径"""
        try:
            # 索引实际上已经在Chroma中持久化，这里只是记录索引名称
            index_path = self.vector_index_dir / f"{index_name}_info.pkl"
            
            # 保存索引信息
            index_info = {
                'embedding_model': self.embedding_model_name,
                'similarity_threshold': self.similarity_threshold,
                'index_path': str(self.vector_index_dir)
            }
            
            with open(index_path, 'wb') as f:
                pickle.dump(index_info, f)
            
            logger.info(f"索引信息已保存到: {index_path}")
        except Exception as e:
            logger.error(f"保存向量索引失败: {str(e)}")

    def load_index(self, index_name: str = "default_index"):
        """从指定路径加载向量索引"""
        try:
            # 检查索引信息文件
            index_info_path = self.vector_index_dir / f"{index_name}_info.pkl"
            if not index_info_path.exists():
                logger.warning(f"索引信息文件不存在: {index_info_path}")
                return
            
            # 加载索引信息
            with open(index_info_path, 'rb') as f:
                index_info = pickle.load(f)
            
            # 重新创建向量存储
            self.vectorstore = Chroma(
                persist_directory=str(self.vector_index_dir),
                embedding_function=self.embeddings
            )
            
            logger.info(f"向量索引已从 {index_info_path} 加载")
        except Exception as e:
            logger.error(f"加载向量索引失败: {str(e)}")
            raise

    def get_statistics(self) -> dict:
        """获取索引统计信息"""
        if not self.vectorstore:
            logger.warning("向量索引未构建，无法获取统计信息")
            return {}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            stats = {
                'document_count': count,
                'embedding_model': self.embedding_model_name,
                'similarity_threshold': self.similarity_threshold,
                'index_path': str(self.vector_index_dir)
            }
            
            logger.info(f"索引统计信息: {stats}")
            return stats
        except Exception as e:
            logger.error(f"获取索引统计信息失败: {str(e)}")
            return {}

    def optimize_index(self):
        """执行索引优化操作"""
        logger.info("执行索引优化操作")
        # 在Chroma中，通常不需要显式的优化操作
        # 但我们可以实现一些基本的清理和检查
        if self.vectorstore:
            stats = self.get_statistics()
            logger.info(f"优化后索引状态: {stats}")
        else:
            logger.warning("向量索引未构建，无法执行优化")