"""
Multi-Agent Research System
===========================
A complete CrewAI-based multi-agent research pipeline.

Agents:
  1. Senior Research Analyst  – gathers recent, credible information (web search)
  2. Research Analyst         – synthesizes findings into insights
  3. Quality Assurance Reviewer – critiques accuracy, bias, completeness
  4. Research Report Writer   – produces the final polished markdown report

Processes supported:
  - sequential  (default): clear pipeline research → analyze → review → write
  - hierarchical: manager agent dynamically delegates and validates

Usage:
  python main.py
  # or
  python main.py --topic "Your research topic" --process hierarchical
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------
load_dotenv()

REQUIRED_KEYS = ["OPENAI_API_KEY", "SERPER_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}\n"
        "Copy .env.example → .env and fill in your keys."
    )

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
search_tool = SerperDevTool(n_results=10)

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
researcher = Agent(
    role="Senior Research Analyst",
    goal="Gather the most relevant, recent, and credible information on the given topic",
    backstory=(
        "You are a meticulous research analyst with over 12 years of experience "
        "in technology and business research. You prioritize primary sources, "
        "recent publications (last 12–18 months preferred), and cross-verify claims. "
        "You extract concrete facts, statistics, trends, and always cite source URLs."
    ),
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
    max_iter=8,
)

analyst = Agent(
    role="Research Analyst & Synthesizer",
    goal="Turn raw research findings into clear, prioritized insights and structured analysis",
    backstory=(
        "You excel at synthesizing large volumes of information into coherent narratives. "
        "You identify trends, contradictions, implications, knowledge gaps, and "
        "structure findings so they are actionable for decision-makers."
    ),
    verbose=True,
    allow_delegation=False,
    max_iter=6,
)

reviewer = Agent(
    role="Quality Assurance Reviewer",
    goal="Critically review the analysis for accuracy, completeness, bias, and clarity",
    backstory=(
        "You are a rigorous peer reviewer. You check for unsupported claims, "
        "missing perspectives, outdated information, logical gaps, and balance. "
        "You provide constructive, specific feedback and a refined version when needed."
    ),
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

writer = Agent(
    role="Research Report Writer",
    goal="Produce a polished, professional, stakeholder-ready research report",
    backstory=(
        "You write clear, executive-ready reports. You use logical structure, "
        "concise language, and a consistent markdown format that is easy to scan "
        "and act upon."
    ),
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def create_tasks(topic: str) -> list[Task]:
    research_task = Task(
        description=(
            f"Conduct thorough research on the topic: **{topic}**\n\n"
            "Focus on the following areas:\n"
            "- Key concepts and precise definitions\n"
            "- Recent developments and breakthroughs (prefer last 12–18 months)\n"
            "- Major players, technologies, frameworks, or approaches\n"
            "- Challenges, risks, limitations, and open problems\n"
            "- Notable real-world applications or case studies\n"
            "- Future outlook and emerging trends\n\n"
            "Use the search tool extensively. Extract concrete facts, numbers, "
            "and dates. Always include source URLs for major claims."
        ),
        expected_output=(
            "A structured research brief containing:\n"
            "- Bullet-point key findings grouped by theme\n"
            "- Source URLs attached to each major claim\n"
            "- Notes on recency and source credibility"
        ),
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            f"Analyze the research findings on **{topic}**.\n\n"
            "Produce a synthesis that includes:\n"
            "- A concise executive summary of the current state\n"
            "- 4–6 prioritized key insights or trends\n"
            "- Important contradictions or debates in the field\n"
            "- Practical implications and opportunities\n"
            "- Knowledge gaps worth further investigation"
        ),
        expected_output=(
            "A clear analytical synthesis with prioritized insights, "
            "supported by evidence from the research brief."
        ),
        agent=analyst,
        context=[research_task],
    )

    review_task = Task(
        description=(
            f"Perform a rigorous quality review of the analysis on **{topic}**.\n\n"
            "Evaluate:\n"
            "- Factual accuracy and source support\n"
            "- Completeness (are important angles missing?)\n"
            "- Bias or one-sided framing\n"
            "- Clarity and logical flow\n"
            "- Actionability of the insights\n\n"
            "Provide specific feedback and a refined version of the analysis."
        ),
        expected_output=(
            "A review report containing:\n"
            "- Strengths of the current analysis\n"
            "- Issues found (with severity)\n"
            "- Concrete improvement suggestions\n"
            "- Final recommended refined analysis"
        ),
        agent=reviewer,
        context=[analysis_task],
    )

    report_task = Task(
        description=(
            f"Write a professional research report on **{topic}** based on the "
            "reviewed and refined analysis.\n\n"
            "Required structure (use Markdown):\n"
            "1. Title\n"
            "2. Executive Summary (3–6 sentences)\n"
            "3. Key Findings\n"
            "4. Detailed Analysis\n"
            "5. Challenges & Risks\n"
            "6. Opportunities & Recommendations\n"
            "7. Sources\n\n"
            "Aim for 800–1500 words. Keep language clear, professional, and scannable."
        ),
        expected_output="A polished, complete Markdown research report ready for stakeholders.",
        agent=writer,
        context=[review_task],
        output_file=str(OUTPUT_DIR / "research_report.md"),
    )

    return [research_task, analysis_task, review_task, report_task]


# ---------------------------------------------------------------------------
# Crew runner
# ---------------------------------------------------------------------------
def run_research(topic: str, process_type: str = "sequential") -> str:
    tasks = create_tasks(topic)

    if process_type.lower() == "hierarchical":
        crew = Crew(
            agents=[researcher, analyst, reviewer, writer],
            tasks=tasks,
            process=Process.hierarchical,
            manager_llm=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
            verbose=True,
            memory=True,
        )
    else:
        crew = Crew(
            agents=[researcher, analyst, reviewer, writer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
        )

    result = crew.kickoff(inputs={"topic": topic})
    return str(result)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System powered by CrewAI"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Research topic (if omitted, you will be prompted)",
    )
    parser.add_argument(
        "--process",
        type=str,
        choices=["sequential", "hierarchical"],
        default="sequential",
        help="Orchestration process (default: sequential)",
    )
    args = parser.parse_args()

    topic = args.topic or input("\nEnter research topic: ").strip()
    if not topic:
        topic = "Multi-agent AI systems in enterprise automation 2025-2026"
        print(f"No topic provided – using default: {topic}")

    print(f"\n🚀 Starting multi-agent research")
    print(f"   Topic   : {topic}")
    print(f"   Process : {args.process}\n")
    print("=" * 70)

    result = run_research(topic, process_type=args.process)

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(result)
    print(f"\n📄 Full report also saved to: {OUTPUT_DIR / 'research_report.md'}")


if __name__ == "__main__":
    main()
