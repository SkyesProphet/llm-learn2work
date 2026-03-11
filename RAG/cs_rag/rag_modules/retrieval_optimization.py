import numpy as np
from typing import List, Any, Tuple, Dict
from rank_bm25 import BM25Okapi
from loguru import logger
from ..config import Config


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""

    def __init__(self, vectorstore=None, chunks: List[Any] = None):
        self.vectorstore = vectorstore
        self.chunks = chunks or []
        self.bm25_retriever = None
        
        # 初始化BM25检索器
        if self.chunks:
            self._initialize_bm25_retriever()
        
        logger.info("检索优化模块初始化完成")

    def _initialize_bm25_retriever(self):
        """初始化BM25检索器"""
        # 提取文档文本用于BM25索引
        texts = [chunk.page_content for chunk in self.chunks if hasattr(chunk, 'page_content')]
        tokenized_texts = [text.split() for text in texts]
        
        # 创建BM25实例
        self.bm25_retriever = BM25Okapi(tokenized_texts)
        
        logger.info(f"BM25检索器初始化完成，索引了 {len(texts)} 个文档块")

    def set_retrievers(self, vectorstore=None, chunks: List[Any] = None):
        """设置向量检索器和BM25检索器"""
        if vectorstore:
            self.vectorstore = vectorstore
        if chunks:
            self.chunks = chunks
            self._initialize_bm25_retriever()
        
        logger.info("检索器设置完成")

    def hybrid_retrieval(self, query: str, k: int = None) -> List[Tuple[Any, float]]:
        """混合检索 - 结合向量检索和BM25检索，使用RRF优化重排"""
        if k is None:
            k = Config.RETRIEVAL_K

        # 执行向量检索
        vector_results = []
        if self.vectorstore:
            try:
                vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
            except Exception as e:
                logger.warning(f"向量检索失败: {str(e)}")

        # 执行BM25检索
        bm25_results = []
        if self.bm25_retriever:
            query_tokens = query.split()
            bm25_scores = self.bm25_retriever.get_scores(query_tokens)
            
            # 获取top-k结果
            top_indices = np.argsort(bm25_scores)[::-1][:k]
            for idx in top_indices:
                if idx < len(self.chunks):
                    score = bm25_scores[idx]
                    bm25_results.append((self.chunks[idx], float(score)))

        # 使用RRF（Reciprocal Rank Fusion）算法融合结果
        fused_results = self._rrf_fusion(vector_results, bm25_results, k)
        
        logger.info(f"混合检索完成，返回 {len(fused_results)} 个结果")
        return fused_results

    def _rrf_fusion(self, vector_results: List[Tuple[Any, float]], 
                   bm25_results: List[Tuple[Any, float]], 
                   k: int = None) -> List[Tuple[Any, float]]:
        """使用RRF算法融合向量检索和BM25检索结果"""
        if k is None:
            k = Config.RETRIEVAL_K

        # 创建文档到排名的映射
        all_docs = set()
        for doc, _ in vector_results + bm25_results:
            all_docs.add(doc)

        # 计算RRF分数
        rrf_scores = {}
        for doc in all_docs:
            rrf_scores[doc] = 0.0

        # 向量检索排名
        for rank, (doc, _) in enumerate(vector_results, 1):
            # 使用倒数排名作为分数，排名越靠前分数越高
            rrf_scores[doc] += 1.0 / (rank + 60)  # 60是RRF公式中的常数

        # BM25检索排名
        for rank, (doc, _) in enumerate(bm25_results, 1):
            rrf_scores[doc] += 1.0 / (rank + 60)

        # 按RRF分数排序并返回top-k
        sorted_docs = sorted(all_docs, key=lambda x: rrf_scores[x], reverse=True)[:k]
        result = [(doc, rrf_scores[doc]) for doc in sorted_docs]

        return result

    def metadata_filtered_retrieval(self, query: str, filters: Dict[str, Any], k: int = None) -> List[Tuple[Any, float]]:
        """带元数据过滤的检索"""
        if k is None:
            k = Config.RETRIEVAL_K

        # 先执行混合检索
        all_results = self.hybrid_retrieval(query, k * 2)  # 获取更多结果以供过滤

        # 应用元数据过滤
        filtered_results = []
        for doc, score in all_results:
            include_doc = True
            
            # 检查所有过滤条件
            for key, value in filters.items():
                if hasattr(doc, 'metadata') and key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            include_doc = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            include_doc = False
                            break
            
            if include_doc:
                filtered_results.append((doc, score))

        # 返回前k个结果
        result = filtered_results[:k]
        
        logger.info(f"元数据过滤检索完成，返回 {len(result)} 个结果")
        return result

    def format_context_for_llm(self, retrieval_results: List[Tuple[Any, float]]) -> str:
        """将检索结果格式化为LLM可读的上下文"""
        context_parts = []
        
        for doc, score in retrieval_results:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                chunk_id = doc.metadata.get('chunk_id', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                
                context_part = f"来源: {source} | 块ID: {chunk_id} | 相关度: {score:.3f}\n{content}\n---\n"
                context_parts.append(context_part)
        
        context_str = "".join(context_parts)
        
        logger.info(f"上下文格式化完成，包含 {len(context_parts)} 个文档块")
        return context_str

    def is_knowledge_sufficient(self, retrieval_results: List[Tuple[Any, float]], query: str) -> bool:
        """判断检索结果是否足以回答问题"""
        if not retrieval_results:
            return False

        # 检查检索结果的相关度分数
        avg_score = sum(score for _, score in retrieval_results) / len(retrieval_results)
        
        # 检查检索到的上下文长度
        total_length = sum(len(doc.page_content) for doc, _ in retrieval_results)
        
        # 判断条件：
        # 1. 平均相关度分数高于阈值
        # 2. 总上下文长度足够
        # 3. 至少有一个结果的相关度较高
        high_score_exists = any(score >= Config.SIMILARITY_THRESHOLD for _, score in retrieval_results)
        
        is_sufficient = (
            avg_score >= Config.SIMILARITY_THRESHOLD * 0.7 and  # 平均分数达到阈值的70%
            total_length >= 100 and  # 总长度至少100字符
            high_score_exists  # 至少有一个高分结果
        )
        
        logger.info(f"知识充分性判断: {'充分' if is_sufficient else '不足'}, "
                   f"平均分数: {avg_score:.3f}, 总长度: {total_length}, 高分存在: {high_score_exists}")
        return is_sufficient

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取检索性能指标"""
        metrics = {
            'vectorstore_available': self.vectorstore is not None,
            'bm25_retriever_available': self.bm25_retriever is not None,
            'indexed_chunks_count': len(self.chunks) if self.chunks else 0
        }
        
        logger.info(f"检索性能指标: {metrics}")
        return metrics