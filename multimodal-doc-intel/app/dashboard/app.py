import streamlit as st
from app.rag.vectorstore import build_vectorstore, query
from langchain_openai import ChatOpenAI
from app.utils.config import get_settings

st.set_page_config(page_title="Industrial Doc Intelligence", page_icon="📄", layout="wide")
st.title("📄 Multimodal Industrial Document Intelligence")

@st.cache_resource
def get_store():
    return build_vectorstore()

store = get_store()
q = st.text_input("Ask about equipment manuals, setpoints, LOTO, IEC requirements...",
                  value="What are the SF6 low pressure actions and isolation requirements?")

if st.button("Search & Answer") and q:
    docs = query(store, q, k=4)
    context = "\n\n".join([f"[{d.metadata.get('source')}]\n{d.page_content[:600]}" for d in docs])
    st.subheader("Retrieved Context")
    st.text(context[:2000])

    settings = get_settings()
    if settings.openai_api_key:
        llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0.1)
        prompt = f"You are an industrial technical assistant. Use only the context below to answer.\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:"
        with st.spinner("Generating answer..."):
            ans = llm.invoke(prompt).content
        st.subheader("Answer")
        st.markdown(ans)
    else:
        st.info("Set OPENAI_API_KEY for LLM answer generation. Showing retrieved chunks only.")

st.caption("Portfolio demo – add real manuals under data/sample_docs.")
