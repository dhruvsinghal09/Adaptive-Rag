import unittest
from unittest.mock import Mock
import importlib
import sys
import types


class ReActAgentRefreshTests(unittest.TestCase):
    def _install_stubs(self):
        langchain_agents = types.ModuleType("langchain.agents")
        langchain_agents.create_react_agent = Mock(side_effect=["agent_a", "agent_b"])
        langchain_agents.AgentExecutor = Mock(side_effect=["exec_a", "exec_b"])

        langchain_core_prompts = types.ModuleType("langchain_core.prompts")

        class _Prompt:
            @staticmethod
            def from_messages(_messages):
                return "prompt_template"

        langchain_core_prompts.ChatPromptTemplate = _Prompt

        config_settings = types.ModuleType("src.config.settings")

        class _Config:
            def prompt(self, _key):
                return "system prompt"

        config_settings.Config = _Config

        llm_openai = types.ModuleType("src.llms.openai")
        llm_openai.llm = Mock(name="llm")

        retriever_setup = types.ModuleType("src.rag.retriever_setup")
        retriever_setup.get_retriever = Mock(side_effect=["tool_a", "tool_b"])

        sys.modules["langchain.agents"] = langchain_agents
        sys.modules["langchain_core.prompts"] = langchain_core_prompts
        sys.modules["src.config.settings"] = config_settings
        sys.modules["src.llms.openai"] = llm_openai
        sys.modules["src.rag.retriever_setup"] = retriever_setup

        return {
            "create_react_agent": langchain_agents.create_react_agent,
            "agent_executor_cls": langchain_agents.AgentExecutor,
            "get_retriever": retriever_setup.get_retriever,
        }

    def test_build_agent_executor_rebuilds_tools_each_call(self):
        stubs = self._install_stubs()
        sys.modules.pop("src.rag.reAct_agent", None)

        module = importlib.import_module("src.rag.reAct_agent")
        build_agent_executor = module.build_agent_executor

        result_a = build_agent_executor()
        result_b = build_agent_executor()

        self.assertEqual(result_a, "exec_a")
        self.assertEqual(result_b, "exec_b")
        self.assertEqual(stubs["get_retriever"].call_count, 2)
        self.assertEqual(stubs["create_react_agent"].call_count, 2)
        self.assertEqual(stubs["agent_executor_cls"].call_count, 2)
        self.assertEqual(stubs["create_react_agent"].call_args_list[0].args[1], ["tool_a"])
        self.assertEqual(stubs["create_react_agent"].call_args_list[1].args[1], ["tool_b"])


if __name__ == "__main__":
    unittest.main()
