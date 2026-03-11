from typing import List, Dict, Any, Optional
from openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from loguru import logger
from ..config import Config


class GenerationIntegrationModule:
    """生成集成模块 - 负责与LLM交互并生成回答"""

    def __init__(self):
        # 初始化大语言模型（接入阿里百炼API）
        self.client = OpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
        )
        
        # 初始化提示词模板
        self.basic_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="基于以下上下文信息回答问题：\n\n{context}\n\n问题：{question}\n\n回答："
        )
        
        self.detailed_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""基于以下上下文信息，详细回答问题并分步骤解释：

上下文信息：
{context}

问题：{question}

请按以下要求回答：
1. 首先总结关键信息
2. 然后给出详细解答
3. 如有必要，提供相关建议

详细回答："""
        )
        
        logger.info("生成集成模块初始化完成")

    def generate_basic_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """生成基础回答"""
        # 构建上下文字符串
        context_str = self._build_context_string(context_docs)
        
        # 构建提示词
        prompt = self.basic_prompt_template.format(context=context_str, question=query)
        
        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            
            answer = response.choices[0].message.content
            logger.info("基础回答生成完成")
            return answer
            
        except Exception as e:
            logger.error(f"生成基础回答失败: {str(e)}")
            return "抱歉，生成回答时出现问题，请稍后再试。"

    def generate_detailed_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """生成详细分步骤回答"""
        # 构建上下文字符串
        context_str = self._build_context_string(context_docs)
        
        # 构建提示词
        prompt = self.detailed_prompt_template.format(context=context_str, question=query)
        
        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content
            logger.info("详细回答生成完成")
            return answer
            
        except Exception as e:
            logger.error(f"生成详细回答失败: {str(e)}")
            return "抱歉，生成详细回答时出现问题，请稍后再试。"

    def rewrite_query(self, original_query: str) -> str:
        """智能查询重写 - 让大模型判断是否需要重写查询"""
        rewrite_prompt = f"""
        分析以下查询，如果查询模糊或不明确，请重写为更精确的查询。
        如果查询已经很明确，则返回原查询。
        
        原查询：{original_query}
        
        请只返回重写后的查询或原查询，不要添加其他文字。
        """
        
        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个查询优化专家，能够判断查询是否需要重写，并进行相应的改写。"},
                    {"role": "user", "content": rewrite_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            logger.info(f"查询重写完成: '{original_query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"查询重写失败: {str(e)}")
            return original_query

    def route_query(self, query: str) -> str:
        """查询路由 - 根据查询类型选择不同的处理方式"""
        routing_prompt = f"""
        分析以下查询的类型，并归类为以下三种之一：
        - list: 列举类问题（如"有哪些..."、"包括什么..."）
        - detail: 详细解释类问题（如"如何..."、"为什么..."、"原理是什么..."）
        - general: 通用类问题（其他类型的查询）
        
        查询：{query}
        
        请只返回分类结果（list、detail或general），不要添加其他文字。
        """
        
        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个查询分类专家，能够准确识别查询的类型。"},
                    {"role": "user", "content": routing_prompt}
                ],
                temperature=0.1,
                max_tokens=20
            )
            
            route_type = response.choices[0].message.content.strip().lower()
            
            # 验证返回的类型是否有效
            if route_type not in ['list', 'detail', 'general']:
                route_type = 'general'  # 默认为通用类型
                
            logger.info(f"查询路由完成: '{query}' -> '{route_type}'")
            return route_type
            
        except Exception as e:
            logger.error(f"查询路由失败: {str(e)}")
            return 'general'

    def _build_context_string(self, context_docs: List[Dict[str, Any]]) -> str:
        """构建上下文字符串"""
        context_parts = []
        
        for doc in context_docs:
            if isinstance(doc, dict):
                content = doc.get('content', '')
                source = doc.get('source', 'Unknown')
                score = doc.get('score', 1.0)
            else:
                # 如果是LangChain Document对象
                content = getattr(doc, 'page_content', '')
                source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
                score = getattr(doc, 'metadata', {}).get('relevance_score', 1.0)
            
            context_part = f"[来源: {source}, 相关度: {score:.3f}]\n{content}\n---\n"
            context_parts.append(context_part)
        
        return "".join(context_parts)

    def stream_response(self, query: str, context_docs: List[Dict[str, Any]], route_type: str = 'general') -> str:
        """流式响应输出"""
        context_str = self._build_context_string(context_docs)
        
        # 根据路由类型选择提示词
        if route_type == 'list':
            prompt_template = self.detailed_prompt_template  # 使用详细模板，但聚焦于列举
            system_msg = Config.SYSTEM_PROMPT + "\n回答时请以列表形式呈现结果。"
        elif route_type == 'detail':
            prompt_template = self.detailed_prompt_template
            system_msg = Config.SYSTEM_PROMPT + "\n请提供详细的分步骤解释。"
        else:  # general
            prompt_template = self.basic_prompt_template
            system_msg = Config.SYSTEM_PROMPT
        
        prompt = prompt_template.format(context=context_str, question=query)
        
        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=768,
                stream=True  # 启用流式输出
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # 在实际应用中，这里可以yield content来实现真正的流式输出
                    # yield content
            
            logger.info("流式响应生成完成")
            return full_response
            
        except Exception as e:
            logger.error(f"流式响应生成失败: {str(e)}")
            return "抱歉，生成响应时出现问题，请稍后再试。"

    def generate_json_response(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成JSON格式响应"""
        context_str = self._build_context_string(context_docs)
        
        json_prompt = f"""
        基于以下上下文信息回答问题，并以JSON格式返回结果：

        上下文信息：
        {context_str}

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
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": Config.SYSTEM_PROMPT + " 请始终以有效的JSON格式返回结果。"},
                    {"role": "user", "content": json_prompt}
                ],
                temperature=0.3,
                max_tokens=512,
                response_format={"type": "json_object"}
            )
            
            import json
            json_response = json.loads(response.choices[0].message.content)
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