from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PromptContract(BaseModel):
    """A versioned, explicit behavioral contract for an LLM component."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    instructions: tuple[str, ...] = Field(min_length=1)
    constraints: tuple[str, ...] = Field(default_factory=tuple)
    failure_behavior: str = Field(min_length=1)
    output_contract: str = Field(min_length=1)

    @field_validator("instructions", "constraints")
    @classmethod
    def _strip_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item.strip())
        if not cleaned and value is not None:
            return cleaned
        return cleaned

    def render_system(self) -> str:
        """Render the contract as a deterministic system instruction."""
        sections = [
            f"PROMPT: {self.name} v{self.version}",
            f"PURPOSE:\n{self.purpose.strip()}",
            "INSTRUCTIONS:\n" + "\n".join(f"{i}. {item}" for i, item in enumerate(self.instructions, 1)),
        ]

        if self.constraints:
            sections.append(
                "CONSTRAINTS:\n"
                + "\n".join(f"- {item}" for item in self.constraints)
            )

        sections.extend(
            [
                f"FAILURE BEHAVIOR:\n{self.failure_behavior.strip()}",
                f"OUTPUT CONTRACT:\n{self.output_contract.strip()}",
            ]
        )
        return "\n\n".join(sections)


class PromptPackage(BaseModel):
    """Reusable LangChain prompt package backed by a PromptContract."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    contract: PromptContract
    human_template: str = Field(min_length=1)

    def system_message(self) -> str:
        return self.contract.render_system()

    def messages(self) -> list[tuple[str, str]]:
        """Return a provider-neutral message specification."""
        return [
            ("system", self.system_message()),
            ("human", self.human_template),
        ]

    def as_langchain_prompt(self):
        """Build the LangChain ChatPromptTemplate only at the framework boundary."""
        from langchain_core.prompts import ChatPromptTemplate

        return ChatPromptTemplate.from_messages(self.messages())
