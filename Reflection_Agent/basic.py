from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
import os
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-60fb2b16-fcdf-4bdd-9558-48836c09e237"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-408fcd10-8c77-4fd2-8256-5648050454ed"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext
from chains import generation_chain, reflection_chain

graph = MessageGraph()

REFLECT = "reflect"
GENERATE = "generate"

def generate_node(state):
    # Pass only initial request + latest critique to avoid context overflow on small models
    messages = [state[0], state[-1]] if len(state) > 1 else state
    return generation_chain.invoke({"messages": messages})

def reflect_node(state):
    # Critique only the latest generated script, not the full history
    messages = [state[0], state[-1]]
    response = reflection_chain.invoke({"messages": messages})
    return [HumanMessage(content=response.content)]

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

def should_continue(state):
    if(len(state) > 4):
        print(f"\n iteration \n")
        return END
    return REFLECT

graph.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

INPUT = "Write a mysterious story about a lost island with a hidden treasure and a secret society guarding it. The story should be engaging, suspenseful, and suitable for a 4-5 minute YouTube video script."

@observe(name="reflection-agent")
def run_agent(message: str) -> str:
    lf = get_client()
    trace_context = TraceContext(
        trace_id=lf.get_current_trace_id(),
        parent_span_id=lf.get_current_observation_id(),
    )
    handler = CallbackHandler(trace_context=trace_context)
    response = app.invoke(
        HumanMessage(content=message),
        config={"callbacks": [handler]},
    )
    return response[-1].content if response else ""

result = run_agent(INPUT)
print(result)