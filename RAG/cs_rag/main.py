import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import Config
from .rag_modules.data_preparation import DataPreparationModule
from .rag_modules.index_construction import IndexConstructionModule
from .rag_modules.retrieval_optimization import RetrievalOptimizationModule
from .rag_modules.generation_integration import GenerationIntegrationModule


class RAGSystem:
    """网络安全知识问答RAG系统主类"""

    def __init__(self):
        # 初始化各核心模块
        self.data_prep_module = DataPreparationModule()
        self.index_module = IndexConstructionModule()
        self.retrieval_module = None  # 在索引构建后初始化
        self.gen_module = GenerationIntegrationModule()
        
        logger.info("RAG系统初始化完成")

    def initialize_system(self):
        """系统初始化：加载配置，初始化各核心模块"""
        logger.info("开始系统初始化...")
        
        # 检查向量索引是否存在，必要时自动构建
        self.check_and_build_index()
        
        logger.info("系统初始化完成")

    def check_and_build_index(self):
        """检查向量索引是否存在，必要时自动构建"""
        index_exists = self._check_index_exists()
        
        if index_exists:
            logger.info("检测到现有向量索引，正在加载...")
            try:
                self.index_module.load_index()
                logger.info("向量索引加载成功")
                
                # 初始化检索优化模块
                chunks, _ = self.data_prep_module.prepare_data()
                self.retrieval_module = RetrievalOptimizationModule(
                    vectorstore=self.index_module.vectorstore,
                    chunks=chunks
                )
                
            except Exception as e:
                logger.error(f"加载现有索引失败: {str(e)}")
                logger.info("正在重新构建索引...")
                self._build_new_index()
        else:
            logger.info("未检测到现有向量索引，正在构建新索引...")
            self._build_new_index()

    def _check_index_exists(self) -> bool:
        """检查向量索引是否存在"""
        index_path = Config.VECTOR_INDEX_DIR
        return index_path.exists() and any(index_path.iterdir())

    def _build_new_index(self):
        """构建新的向量索引"""
        try:
            # 准备数据
            logger.info("正在准备数据...")
            chunks, metadata = self.data_prep_module.prepare_data()
            
            # 构建向量索引
            logger.info("正在构建向量索引...")
            self.index_module.build_vector_index(chunks)
            
            # 保存索引
            self.index_module.save_index()
            
            # 初始化检索优化模块
            self.retrieval_module = RetrievalOptimizationModule(
                vectorstore=self.index_module.vectorstore,
                chunks=chunks
            )
            
            logger.info("新索引构建完成")
            
        except Exception as e:
            logger.error(f"构建新索引失败: {str(e)}")
            raise

    def query(self, user_query: str) -> str:
        """问答接口：提供 query() 方法供外部调用"""
        try:
            logger.info(f"收到查询: {user_query}")
            
            # 1. 智能查询重写
            rewritten_query = self.gen_module.rewrite_query(user_query)
            if rewritten_query != user_query:
                logger.info(f"查询已重写: {user_query} -> {rewritten_query}")
            
            # 2. 查询路由
            route_type = self.gen_module.route_query(rewritten_query)
            logger.info(f"查询路由类型: {route_type}")
            
            # 3. 执行混合检索
            retrieval_results = self.retrieval_module.hybrid_retrieval(rewritten_query)
            
            # 4. 检查知识充分性
            if not self.retrieval_module.is_knowledge_sufficient(retrieval_results, rewritten_query):
                return "对不起，暂时不具备相关的知识，请重新提问，或更新知识库。"
            
            # 5. 格式化上下文
            context_str = self.retrieval_module.format_context_for_llm(retrieval_results)
            
            # 6. 根据路由类型生成回答
            if route_type == 'detail':
                answer = self.gen_module.generate_detailed_answer(rewritten_query, retrieval_results)
            elif route_type == 'list':
                # 对于列表类问题，使用基础回答但强调列举
                answer = self.gen_module.generate_basic_answer(rewritten_query, retrieval_results)
            else:  # general
                answer = self.gen_module.generate_basic_answer(rewritten_query, retrieval_results)
            
            logger.info("查询处理完成")
            return answer
            
        except Exception as e:
            logger.error(f"处理查询时发生错误: {str(e)}")
            return "对不起，处理您的查询时发生了错误，请稍后再试。"

    def run_interactive_mode(self):
        """交互模式：提供命令行交互式问答界面"""
        logger.info("启动交互模式")
        print("\n=== 网络安全知识问答系统 ===")
        print("输入 'quit' 或 'exit' 退出系统\n")
        
        while True:
            try:
                user_input = input("请输入您的问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用，再见！")
                    break
                
                if not user_input:
                    continue
                
                # 处理查询
                response = self.query(user_input)
                print(f"\n回答: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                logger.error(f"交互模式中发生错误: {str(e)}")
                print("系统出现错误，请稍后再试")


def main():
    """主函数"""
    try:
        # 创建RAG系统实例
        rag_system = RAGSystem()
        
        # 初始化系统
        rag_system.initialize_system()
        
        # 启动交互模式
        rag_system.run_interactive_mode()
        
    except Exception as e:
        logger.error(f"系统运行错误: {str(e)}")
        print(f"系统运行错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()