import streamlit as st
import requests
import json

st.title("🤖 Multi-Agent Research Assistant")
st.markdown("*Ask a question and watch the agents work in real-time!*")

query = st.text_input("Ask a question:", placeholder="e.g., What is LangGraph?")

if st.button("🔍 Ask", type="primary"):
    if not query:
        st.warning("Please enter a question!")
    else:
        # Create placeholders for dynamic updates
        progress_container = st.container()
        answer_container = st.container()
        
        with progress_container:
            st.subheader("🔄 Agent Progress")
            manager_status = st.empty()
            research_status = st.empty()
            validation_status = st.empty()
            summary_status = st.empty()
        
        with answer_container:
            st.subheader("📝 Answer")
            answer_placeholder = st.empty()
        
        # Stream the response
        try:
            response = requests.get(
                "http://127.0.0.1:8000/chat/stream",
                params={"query": query},
                stream=True
            )
            
            final_answer = ""
            logs = []
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        
                        event_type = data.get('type')
                        
                        if event_type == 'start':
                            st.toast(data.get('message', 'Starting...'))
                        
                        elif event_type == 'agent':
                            agent = data.get('agent')
                            status = data.get('status')
                            message = data.get('message', '')
                            
                            if agent == 'Manager':
                                if status == 'running':
                                    manager_status.info("🔄 Manager Agent: Creating plan...")
                                elif status == 'complete':
                                    manager_status.success(f"✅ Manager Agent: {message}")
                            
                            elif agent == 'Research':
                                if status == 'running':
                                    research_status.info("🔄 Research Agent: Searching web...")
                                elif status == 'complete':
                                    research_status.success(f"✅ Research Agent: {message}")
                            
                            elif agent == 'Validation':
                                if status == 'running':
                                    validation_status.info("🔄 Validation Agent: Validating facts...")
                                elif status == 'complete':
                                    validation_status.success(f"✅ Validation Agent: {message}")
                            
                            elif agent == 'Summary':
                                if status == 'running':
                                    summary_status.info("🔄 Summary Agent: Generating summary...")
                                elif status == 'complete':
                                    summary_status.success(f"✅ Summary Agent: {message}")
                        
                        elif event_type == 'summary_start':
                            final_answer = ""
                        
                        elif event_type == 'summary_chunk':
                            chunk = data.get('content', '')
                            final_answer += chunk
                            answer_placeholder.markdown(final_answer)
                        
                        elif event_type == 'complete':
                            final_answer = data.get('final_answer', '')
                            logs = data.get('logs', [])
                            answer_placeholder.markdown(final_answer)
                            
                            # Show logs in expander
                            with st.expander("📋 View Agent Logs"):
                                for log in logs:
                                    st.text(f"• {log}")
                            
                            st.success("✨ Research complete!")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("💡 Tip: Make sure the backend server is running on http://127.0.0.1:8000")
