import streamlit as st
import asyncio
import os
import json
from core import Manager

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Document Intelligence Engine",
    layout="wide"
)

st.title("📄 AI Document Intelligence Engine")

# -------------------------------------------------
# Session State
# -------------------------------------------------

if "manager" not in st.session_state:
    st.session_state.manager = Manager(
        model="qwen3-nothink:latest"
    )

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.header("Configuration")

work_type_map = {
    "Resume": 1,
    "Invoice": 2,
    "Email": 3
}

selected_type = st.sidebar.selectbox(
    "Select Document Type",
    list(work_type_map.keys())
)

work_type = work_type_map[selected_type]

# -------------------------------------------------
# Upload
# -------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["txt", "md"],
    accept_multiple_files=True
)

# -------------------------------------------------
# Worker
# -------------------------------------------------

async def process_file(manager, work_type, uploaded_file):

    try:
        content = uploaded_file.read().decode("utf-8")

        result = await manager.chat(work_type, content)

        if result is None:
            return {
                "name": uploaded_file.name,
                "status": "error",
                "message": "Extraction returned None"
            }

        return {
            "name": uploaded_file.name,
            "status": "success",
            "data": result
        }

    except Exception as e:
        return {
            "name": uploaded_file.name,
            "status": "error",
            "message": str(e)
        }

# -------------------------------------------------
# Runner
# -------------------------------------------------

async def run_processing(manager, work_type, uploaded_files,
                         progress_bar, status_text):

    os.makedirs("out", exist_ok=True)

    tasks = [
        asyncio.create_task(
            process_file(manager, work_type, file)
        )
        for file in uploaded_files
    ]

    results = []

    total = len(tasks)

    completed = 0

    for task in asyncio.as_completed(tasks):

        result = await task

        completed += 1

        status_text.info(
            f"Processed {completed}/{total} : {result['name']}"
        )

        if result["status"] == "success":

            filename = os.path.splitext(result["name"])[0]

            output_path = os.path.join(
                "out",
                f"{filename}.json"
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    result["data"].model_dump(),
                    f,
                    indent=4,
                    ensure_ascii=False
                )

        results.append(result)

        progress_bar.progress(completed / total)

    status_text.success("✅ Processing Complete!")

    return results

# -------------------------------------------------
# UI
# -------------------------------------------------

if st.button("Process Documents"):

    if not uploaded_files:
        st.warning("Please upload at least one document.")

    else:

        progress_bar = st.progress(0)

        status_text = st.empty()

        results = asyncio.run(
            run_processing(
                st.session_state.manager,
                work_type,
                uploaded_files,
                progress_bar,
                status_text
            )
        )

        st.divider()

        st.subheader("Results")

        for result in results:

            if result["status"] == "success":

                st.success(f"✅ {result['name']}")

                with st.expander(result["name"]):

                    st.json(
                        result["data"].model_dump()
                    )

            else:

                st.error(
                    f"❌ {result['name']}\n\n{result['message']}"
                )
