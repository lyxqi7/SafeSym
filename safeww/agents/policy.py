from __future__ import annotations

from dataclasses import dataclass
from PIL import Image

from .actions import AgentAction, parse_agent_action
from .llm import ChatClient, multimodal_user_content
from .prompts import BASELINE_PROMPT, PLANNING_PROMPT, SAFETY_PLANNING_PROMPT


AgentKind = str


@dataclass(frozen=True)
class PolicyDecision:
    raw_output: str
    action: AgentAction | None


@dataclass
class WebAgentPolicy:
    kind: AgentKind
    llm: ChatClient

    def decide(
        self,
        task: str,
        obs_text: str,
        image: Image.Image | None,
        history: list[str],
        last_obs: str | None = None,
        plan: str = "",
        last_error: str | None = None,
        safety_context: str = "",
        url: str | None = None,
    ) -> PolicyDecision:
        if self.kind == "baseline":
            system_prompt = BASELINE_PROMPT
        elif self.kind == "planning":
            system_prompt = PLANNING_PROMPT
        elif self.kind == "safety_planning":
            system_prompt = SAFETY_PLANNING_PROMPT
        else:
            raise ValueError(f"Unknown agent kind: {self.kind}")

        safety_section = f"""
SAFETY STATUS:
{safety_context}
""" if safety_context else ""

        user_prompt = f"""TASK:
{task}

PLAN:
{plan}
{safety_section}

OBS:
{obs_text}

PRE OBS:
{last_obs or ""}

LAST ERROR:
{last_error or ""}

HISTORY:
{history}
"""
        output = self.llm.chat(system_prompt, multimodal_user_content(user_prompt, image))
        return PolicyDecision(raw_output=output, action=parse_agent_action(output, obs_text))

    def next_action(
        self,
        task: str,
        obs_text: str,
        image: Image.Image | None,
        history: list[str],
        last_obs: str | None = None,
        plan: str = "",
        last_error: str | None = None,
        safety_context: str = "",
        url: str | None = None,
    ) -> AgentAction | None:
        return self.decide(
            task=task,
            obs_text=obs_text,
            image=image,
            history=history,
            last_obs=last_obs,
            plan=plan,
            last_error=last_error,
            safety_context=safety_context,
            url=url,
        ).action
