from typing import List, Dict, Any, Optional, Tuple, Callable
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from loguru import logger
from ..config import Config
from ..rag_modules.retrieval_optimization import RetrievalOptimizationModule


class QueryRewriterTool(BaseTool):
    """查询重写工具"""
    name: str = "query_rewriter"
    description: str = "当用户查询模糊或不明确时，重写为更精确的查询"
    return_direct: bool = True
    
    def _run(self, query: str) -> str:
        """重写查询"""
        rewrite_prompt = f"""
        分析以下查询，如果查询模糊或不明确，请重写为更精确的查询。
        如果查询已经很明确，则返回原查询。
        
        原查询：{query}
        
        请只返回重写后的查询或原查询，不要添加其他文字。
        """
        
        try:
            llm = ChatOpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                model=Config.LLM_MODEL_NAME,
                temperature=0.1
            )
            
            response = llm.invoke([
                SystemMessage(content="你是一个查询优化专家，能够判断查询是否需要重写，并进行相应的改写。"),
                HumanMessage(content=rewrite_prompt)
            ])
            
            rewritten_query = response.content.strip()
            logger.info(f"查询重写完成: '{query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"查询重写失败: {str(e)}")
            return query


class QueryRouterTool(BaseTool):
    """查询路由工具"""
    name: str = "query_router"
    description: str = "根据查询类型选择不同的处理方式"
    return_direct: bool = True
    
    def _run(self, query: str) -> str:
        """路由查询"""
        routing_prompt = f"""
        分析以下查询的类型，并归类为以下三种之一：
        - list: 列举类问题（如\"有哪些...\", \"包括什么...\"）
        - detail: 详细解释类问题（如\"如何...\", \"为什么...\", \"原理是什么...\"）
        - general: 通用类问题（其他类型的查询）
        
        查询：{query}
        
        请只返回分类结果（list、detail或general），不要添加其他文字。
        """
        
        try:
            llm = ChatOpenAI(
                api_key=Config.LLMM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                model=Config.LLM_MODEL_NAME,
                temperature=0.1
            )
            
            response = llm.invoke([
                SystemMessage(content="你是一个查询分类专家，能够准确识别查询的类型。"),
                HumanMessage(content=routing_prompt)
            ])
            
            route_type = response.content.strip().lower()
            
            # 验证返回的类型是否有效
            if route_type not in ['list', 'detail', 'general']:
                route_type = 'general'  # 默认为通用类型
                
            logger.info(f"查询路由完成: '{query}' -> '{route_type}'")
            return route_type
            
        except Exception as e:
            logger.error(f"查询路由失败: {str(e)}")
            return 'general'


class ContextRetrieverTool(BaseTool):
    """上下文检索工具"""
    name: str = "context_retriever"
    description: str = "根据查询检索相关的上下文信息"
    return_direct: bool = True
    retrieval_module: RetrievalOptimizationModule
    
    def __init__(self, retrieval_module: RetrievalOptimizationModule):
        super().__init__()
        self.retrieval_module = retrieval_module

    def _run(self, query: str, route_type: str = 'general') -> List[Tuple[Any, float]]:
        """检索上下文"""
        logger.info(f"正在检索与 '{query}' 相关的上下文")
        
        # 执行混合检索
        retrieval_results = self.retrieval_module.hybrid_retrieval(query)
        
        # 检查知识充分性
        if not self.retrieval_module.is_knowledge_sufficient(retrieval_results, query):
            logger.warning("检索到的知识不足以回答问题")
            return []
        
        return retrieval_results


class AnswerValidatorTool(BaseTool):
    """回答验证工具"""
    name: str = "answer_validator"
    description: str = "验证生成的回答是否准确、完整、可靠"
    return_direct: bool = True
    
    def _run(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """验证回答"""
        validation_prompt = f"""
        请评估以下回答的质量和可靠性：

        问题：{query}

        回答：{answer}

        上下文：{context}

        请从以下几个方面进行评估：
        1. 准确性：回答是否准确反映了上下文中的信息？
        2. 完整性：回答是否完整地回答了问题？
        3. 相关性：回答是否与问题相关？
        4. 可靠性：回答是否可靠，有无误导性信息？

        请返回以下JSON格式：
        {{
          "is_valid": true/false,
          "confidence_score": 0-1的分数,
          "feedback": "改进建议",
          "needs_requery": true/false
        }}
        """
        
        try:
            llm = ChatOpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                model=Config.LLM_MODEL_NAME,
                temperature=0.2
            )
            
            response = llm.invoke([
                SystemMessage(content="你是一个回答质量评估专家，能够准确评估回答的质量和可靠性。"),
                HumanMessage(content=validation_prompt)
            ])
            
            import json
            validation_result = json.loads(response.content)
            logger.info(f"回答验证完成，置信度: {validation_result.get('confidence_score', 0)}")
            return validation_result
            
        except Exception as e:
            logger.error(f"回答验证失败: {str(e)}")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "feedback": "验证过程中出错",
                "needs_requery": True
            }


class GenerationIntegrationModule:
    """生成集成模块 - 负责与LLM交互并生成回答，实现多步工作流和自主思考"""

    def __init__(self, retrieval_module: RetrievalOptimizationModule):
        self.retrieval_module = retrieval_module
        self.tools = [
            QueryRewriterTool(),
            QueryRouterTool(),
            ContextRetrieverTool(retrieval_module),
            AnswerValidatorTool()
        ]
        
        # 创建LangChain Agent
        self.agent = self._create_agent()
        
        logger.info("生成集成模块初始化完成")

    def _create_agent(self):
        """创建LangChain Agent"""
        # 定义系统提示
        system_prompt = f"""
        你是一个网络安全专家，使用RAG系统回答用户问题。你的工作流程如下：
        1. 首先理解用户查询，必要时重写查询
        2. 根据查询类型选择合适的处理方式
        3. 检索相关上下文信息
        4. 生成回答
        5. 验证回答的准确性和可靠性
        6. 如有必要，重新查询或重写查询
        
        {Config.SYSTEM_PROMPT}
        """
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 创建LLM
        llm = ChatOpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            model=Config.LLM_MODEL_NAME,
            temperature=0.4
        )
        
        # 创建Agent
        agent = create_openai_tools_agent(llm, self.tools, prompt)
        
        # 创建Agent执行器
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor

    def generate_answer(self, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """生成回答，实现多步工作流"""
        logger.info(f"开始处理查询: {query}")
        
        try:
            # 准备输入
            inputs = {
                "input": query
            }
            
            if chat_history:
                inputs["chat_history"] = chat_history
            
            # 执行Agent
            result = self.agent.invoke(inputs)
            
            # 处理结果
            answer = result["output"]
            
            # 提取上下文信息（从Agent的中间步骤获取）
            context = self._extract_context_from_agent(result)
            
            logger.info("回答生成完成")
            return {
                "answer": answer,
                "context": context,
                "query": query,
                "route_type": self._detect_route_type(query)
            }
            
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return {
                "answer": "抱歉，生成回答时出现问题，请稍后再试。",
                "context": "",
                "query": query,
                "route_type": "general"
            }

    def _extract_context_from_agent(self, agent_result: Dict) -> str:
        """从Agent执行结果中提取上下文"""
        # 这里需要根据实际的Agent执行结果结构提取上下文
        # 实际实现可能需要更复杂的逻辑
        return "检索到的相关上下文信息..."

    def _detect_route_type(self, query: str) -> str:
        """检测查询类型"""
        try:
            # 使用QueryRouterTool来确定路由类型
            llm = ChatOpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                model=Config.LLM_MODEL_NAME,
                temperature=0.1
            )
            
            routing_prompt = f"""
            分析以下查询的类型，并归类为以下三种之一：
            - list: 列举类问题（如\"有哪些...\", \"包括什么...\"）
            - detail: 详细解释类问题（如\"如何...\", \"为什么...\", \"原理是什么...\"）
            - general: 通用类问题（其他类型的查询）
            
            查询：{query}
            
            请只返回分类结果（list、detail或general），不要添加其他文字。
            """
            
            response = llm.invoke([
                SystemMessage(content="你是一个查询分类专家，能够准确识别查询的类型。"),
                HumanMessage(content=routing_prompt)
            ])
            
            route_type = response.content.strip().lower()
            
            if route_type not in ['list', 'detail', 'general']:
                route_type = 'general'
                
            return route_type
            
        except:
            return 'general'

    def stream_response(self, query: str, chat_history: Optional[List] = None) -> str:
        """流式响应输出"""
        logger.info(f"开始流式处理查询: {query}")
        
        # 使用基础回答生成方法进行流式输出
        # 注意：LangChain Agent本身不直接支持流式输出，这里简化实现
        try:
            # 先获取检索结果
            context_retriever = ContextRetrieverTool(self.retrieval_module)
            retrieval_results = context_retriever._run(query)
            
            if not retrieval_results:
                return "对不起，暂时不具备相关的知识，请重新提问，或更新知识库。"

            # 构建上下文字符串
            context_str = self._build_context_string(retrieval_results)
            
            # 确定路由类型
            route_type = self._detect_route_type(query)
            
            # 根据路由类型构建提示
            if route_type == 'list':
                prompt_template = """基于以下上下文信息回答问题，以列表形式呈现结果：

{context}

问题：{question}

回答："""
            elif route_type == 'detail':
                prompt_template = """基于以下上下文信息，详细回答问题并分步骤解释：

{context}

问题：{question}

请按以下要求回答：
1. 首先总结关键信息
2. 然后给出详细解答
3. 如有必要，提供相关建议

详细回答："""
            else:  # general
                prompt_template = """基于以下上下文信息回答问题：

{context}

问题：{question}

回答："""

            prompt = prompt_template.format(context=context_str, question=query)
            
            # 创建流式响应的LLM
            llm = ChatOpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                model=Config.LLM_MODEL_NAME,
                temperature=0.4,
                max_tokens=768,
                streaming=True
            )
            
            # 流式生成响应
            full_response = ""
            for chunk in llm.stream([
                SystemMessage(content=Config.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]):
                if chunk.content:
                    full_response += chunk.content
                    # 在实际应用中，这里可以yield chunk.content
                    # yield chunk.content
            
            logger.info("流式响应生成完成")
            return full_response
            
        except Exception as e:
            logger.error(f"流式响应生成失败: {str(e)}")
            return "抱歉，生成响应时出现问题，请稍后再试。"

    def _build_context_string(self, context_docs: List[Tuple[Any, float]]) -> str:
        """构建上下文字符串"""
        context_parts = []
        
        for doc, score in context_docs:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                chunk_id = doc.metadata.get('chunk_id', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                
                context_part = f"[来源: {source}, 块ID: {chunk_id}, 相关度: {score:.3f}]\n{content}\n---\n"
                context_parts.append(context_part)
        
        return "".join(context_parts)

    def generate_json_response(self, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """生成JSON格式响应"""
        result = self.generate_answer(query, chat_history)
        
        if not result.get('context'):
            return {
                "answer": "对不起，暂时不具备相关的知识，请重新提问，或更新知识库。",
                "confidence": 0.0,
                "sources": [],
                "related_topics": []
            }

        # 使用LLM生成结构化JSON响应
        json_prompt = f"""
        基于以下上下文信息回答问题，并以JSON格式返回结果：

        上下文信息：
        {result['context']}

        问题：{query}

        请返回以下JSON格式：
        {{
          "answer": "答案内容",
          "confidence": "置信度分数(0-1)",
          "sources": ["来源文档列表"],
          "related_topics": ["相关主题列表"]
        }}
        """
        
        try:
            llm = ChatOpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                model=Config.LLM_MODEL_NAME,
                temperature=0.3
            )
            
            response = llm.invoke([
                SystemMessage(content=Config.SYSTEM_PROMPT + " 请始终以有效的JSON格式返回结果。"),
                HumanMessage(content=json_prompt)
            ])
            
            import json
            json_response = json.loads(response.content)
            logger.info("JSON格式响应生成完成")
            return json_response
            
        except Exception as e:
            logger.error(f"JSON格式响应生成失败: {str(e)}")
            return {
                "answer": "抱歉，生成JSON格式响应时出现问题",
                "confidence": 0.0,
                "sources": [],
                "related_topics": []
            }