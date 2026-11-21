from discussion_intelligence.llm_synthesis import create_gemini_provider
from discussion_intelligence.reports import Evidence

evidence = [
    Evidence(
        id="reddit:comment_101",
        conversation_id="reddit:thread_1",
        source_id="comment_101",
        parent_id=None,
        speaker_id="user_1",
        text="AI helps me understand difficult concepts.",
        text_clean="AI helps me understand difficult concepts.",
    ),
    Evidence(
        id="reddit:comment_102",
        conversation_id="reddit:thread_1",
        source_id="comment_102",
        parent_id=None,
        speaker_id="user_2",
        text="Students depend on AI too much.",
        text_clean="Students depend on AI too much.",
    ),
]

provider = create_gemini_provider()

try:
    result = provider.synthesize(
        evidence,
        query="What do students think about AI in education?",
    )

    print("Gemini synthesis successful!")

    for point in result.points:
        print("Quotation:", point.text)
        print("Evidence IDs:", point.evidence_ids)

except Exception as error:
    print("Gemini synthesis failed:", type(error).__name__, str(error))
