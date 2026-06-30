from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a YouTube mystery channel script writer for faceless videos. "
            "Write mystery scripts about islands, planets, or tech gadget evolution. "
            "Every script must: open with a strong hook in the first 30 seconds, build suspense steadily, "
            "hide a reveal until near the end, then deliver it with impact. "
            "Target exactly 600-750 words — this is narrated over AI visuals on higgsfield.ai for a 4-5 minute video. "
            "Output ONLY the narration script, no stage directions or scene labels. "
            "If the previous message is a critique, address every point raised in it and rewrite the script from scratch."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert viral YouTube content coach. "
            "You will receive a mystery video script. Critique ONLY that script — do not reference any prior versions. "
            "Your feedback must cover: "
            "1. Hook (first 30 seconds) — is it strong enough to stop a scroll? "
            "2. Suspense pacing — does tension build consistently? "
            "3. Reveal — is it surprising, well-timed, and satisfying? "
            "4. Word count — is it within 600-750 words for a 4-5 min video? "
            "5. Viral potential — title-worthy moments, emotional pull, rewatch value. "
            "Be specific: quote the weak lines and suggest exact rewrites."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOllama(model="ornith:9b")

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

