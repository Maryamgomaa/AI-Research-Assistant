import json


class PromptTemplates:
    @staticmethod
    def report_system_prompt() -> str:
        return (
            "You are an expert academic research analyst specialising in NLP, "
            "deep learning, and Arabic language processing. "
            "You read research papers and produce structured, precise summaries. "
            "You ALWAYS respond with valid JSON only — no markdown, no preamble, "
            "no trailing text. If information is not available in the text, "
            "use the string 'Not specified' for that field."
        )

    @staticmethod
    def report_generation_prompt(
        title: str,
        authors: list,
        abstract: str,
        text_excerpt: str,
        topic: str = "",
    ) -> str:
        authors_str = ", ".join(authors[:5]) + ("..." if len(authors) > 5 else "")
        topic_context = (
            f"\nThe researcher is studying: '{topic}'\n" if topic else ""
        )

        return f"""Analyse the following academic paper and return a JSON object with EXACTLY these fields:{topic_context}

Paper Title: {title}
Authors: {authors_str}
Abstract: {abstract}

Text excerpt:
{text_excerpt}

Return JSON with these exact keys:
{{
  "title": "Full paper title",
  "authors": ["Author 1", "Author 2"],
  "problem": "What specific research problem does this paper address?",
  "methodology": "What methods, architectures, or algorithms are proposed?",
  "attention_mechanism": "What attention mechanism is used (if any)? Self-attention, cross-attention, multi-head, sparse, etc.",
  "dataset": "What datasets are used for training and/or evaluation?",
  "results": "Key quantitative results and benchmark scores",
  "contributions": "Main contributions listed clearly (2-4 bullet points as a string)",
  "limitations": "Acknowledged limitations or weaknesses",
  "arabic_nlp_relevance": "How is this paper relevant to Arabic NLP research? Rate: High / Medium / Low and explain.",
  "future_research": "Suggested future research directions based on this paper",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Output ONLY valid JSON. No explanation."""

    @staticmethod
    def rag_system_prompt(language: str = "en") -> str:
        if language == "ar":
            return (
                "أنت مساعد بحثي أكاديمي متخصص في معالجة اللغة الطبيعية وتعلم الآلة. "
                "أجب على الأسئلة بناءً على السياق المقدم فقط. "
                "اذكر المصادر بشكل صريح باستخدام معرّف arXiv. "
                "إذا لم تجد الإجابة في السياق، قل ذلك بوضوح. "
                "لا تختلق معلومات غير موجودة في النصوص المقدمة."
            )
        return (
            "You are an academic research assistant specialised in NLP and machine learning. "
            "Answer questions based ONLY on the provided context passages. "
            "Cite sources explicitly using their arXiv ID (e.g. [arXiv:2301.00234]). "
            "If the context does not contain enough information, state that clearly. "
            "Do NOT fabricate information not present in the provided context."
        )

    @staticmethod
    def rag_prompt(question: str, context: str, language: str = "en") -> str:
        if language == "ar":
            return f"""السياق من الأوراق البحثية:
{context}

السؤال: {question}

الإجابة (استند فقط إلى السياق أعلاه، اذكر المصادر برقم arXiv):"""

        return f"""Context from research papers:
{context}

Question: {question}

Answer (based only on the context above, cite sources by arXiv ID):"""

    @staticmethod
    def topic_summary_prompt(topic: str, paper_reports: list[dict]) -> str:
        report_blocks = []
        for report in paper_reports:
            title = report.get("title", "Unknown title")
            arxiv_id = report.get("arxiv_id", "")
            problem = report.get("problem", "Not specified")
            methodology = report.get("methodology", "Not specified")
            results = report.get("results", "Not specified")
            contributions = report.get("contributions", "Not specified")
            report_blocks.append(
                f"Title: {title} (arXiv:{arxiv_id})\nProblem: {problem}\nMethodology: {methodology}\nResults: {results}\nContributions: {contributions}\n"
            )

        report_summary = "\n\n".join(report_blocks)
        return f"""You are an expert academic research analyst.
You have been given structured summaries for several research papers on the topic: {topic}

Review the paper summaries and produce a combined topic-level report.

Paper summaries:
{report_summary}

Return JSON only with these keys:
{{
  "topic": "The research topic",
  "overall_summary": "Concise summary of the most important findings across all papers",
  "key_insights": ["Top insight 1", "Top insight 2", "Top insight 3"],
  "recommended_papers": ["arXiv:XXXX.XXXX", "arXiv:YYYY.YYYY"],
  "research_trends": "Main trends and themes observed across the papers",
  "next_steps": "Practical next research directions or applications"
}}"""

    @staticmethod
    def query_expansion_prompt(topic: str) -> str:
        return f"""Given the research topic: \"{topic}\"

Generate 3 optimised arXiv search queries that will find the most relevant recent papers.
Consider synonyms, related concepts, and common terminology in NLP/ML research.

Return JSON only:
{{
  "queries": [
    "query 1 using arXiv search syntax",
    "query 2 alternative formulation",
    "query 3 broader/related angle"
  ],
  "english_translation": "English translation if original was in Arabic, else null",
  "key_concepts": ["concept1", "concept2", "concept3"]
}}"""

    @staticmethod
    def relevance_scoring_prompt(topic: str, title: str, abstract: str) -> str:
        return f"""Research topic: \"{topic}\"

Paper title: {title}
Abstract: {abstract}

Score how relevant this paper is to the research topic.
Return JSON only:
{{
  "relevance_score": 0-10 (10 being most relevant),
  "reasoning": "Brief explanation of the score",
  "key_matches": ["matching concept 1", "matching concept 2"]
}}"""