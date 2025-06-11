import gradio as gr
from load import process_pdf, embeddings
from vectorstore import get_retriever
from llm import get_qa_chain, query_llm
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def handle_query(pdf_files, user_query):
    if not pdf_files:
        return "Please upload at least one PDF file."
    all_texts = []
    for pdf_file in pdf_files:
        texts, _ = process_pdf(pdf_file.name)
        all_texts.extend(texts)
    retriever = get_retriever(all_texts, embeddings)
    qa_chain = get_qa_chain(retriever)
    return query_llm(user_query, qa_chain)

def handle_report(pdf_files):
    if not pdf_files:
        return None  # Gradio File output expects None for no file
    all_texts = []
    for pdf_file in pdf_files:
        texts, _ = process_pdf(pdf_file.name)
        all_texts.extend(texts)
    retriever = get_retriever(all_texts, embeddings)
    qa_chain = get_qa_chain(retriever)
    report_prompt = (
        """You are a municipal bond underwriter at Build America Mutual. Your task is to evaluate whether a given municipal bond issuance should be insured. Use the following criteria and context to guide your analysis:Core Evaluation Criteria:

            Credit Stability & Liquidity Benefits

            Does this bond provide investors with long-term, stable cash flow?

            Would insurance improve pricing transparency and reduce volatility or downgrade risk?

            Issuer Characteristics

            Is the issuer small, infrequent, or less transparent (e.g. rural utilities, small towns)?

            Would insurance provide enhanced credibility, particularly for issuers with limited public financial disclosures?

            Sector & Market Conditions

            Is the bond in a sector vulnerable to economic or pandemic-related volatility (e.g. tourism, public transit, higher education)?

            Is the sector currently under fiscal stress or undergoing recovery (e.g. post-pandemic recovery, declining enrollment in universities)?

            Market Demand & Investor Confidence

            Could insurance expand the investor base or attract international buyers?

            Would the credit rating uplift (e.g. from Baa3 to AA) provide meaningful borrowing cost savings?

            Bond Characteristics

            Is the issuance taxable or tax-exempt?

            Does the bond carry a Green or sustainability label (e.g. BAM GreenStar)?

            Macroeconomic & Structural Risks

            Does the issuer face long-term pressures from inflation, expiring stimulus, or climate-related risks (e.g. flooding, wildfire, economic disruption)?

            Is the issuance aligned with infrastructure investment priorities or government-backed initiatives?

            Historical Outperformance

            In similar previous issuances, did insured bonds outperform in terms of credit spreads or secondary market stability?

            Instructions:
            Given a bond description, assess it across these dimensions and provide a clear recommendation:

            ‘Insure’, with explanation of added value, or

            ‘Do Not Insure’, with rationale.
            Include any red flags or exceptional strengths."""
    )
    report, citations = query_llm(report_prompt, qa_chain)
    pdf_path = create_pdf_report(report, citations)
    return pdf_path

def wrap_text(text, width, c, font_name="Helvetica", font_size=11):
    """Wrap text for the given width using the canvas font settings."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if stringWidth(test_line, font_name, font_size) <= width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def create_pdf_report(report_text, citations):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp.name, pagesize=A4)
    width, height = A4
    left_margin = 40
    right_margin = 40
    usable_width = width - left_margin - right_margin
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, y, "Bond Due Diligence Report")
    y -= 32
    c.setFont("Helvetica", 11)

    # Write report text with wrapping
    for line in report_text.split('\n'):
        wrapped = wrap_text(line, usable_width, c)
        for wline in wrapped:
            c.drawString(left_margin, y, wline)
            y -= 15
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)
        y -= 2

    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(left_margin, y, "Citations")
    y -= 20
    c.setFont("Helvetica", 9)

    # Write citations with wrapping
    for cite in citations:
        for line in cite.split('\n'):
            wrapped = wrap_text(line, usable_width, c, font_size=9)
            for wline in wrapped:
                c.drawString(left_margin, y, wline)
                y -= 12
                if y < 60:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 9)
        y -= 8

    c.save()
    return temp.name

def search_documents(pdf_files, keyword):
    if not pdf_files:
        return "Please upload at least one PDF file."
    results = []
    for pdf_file in pdf_files:
        texts, _ = process_pdf(pdf_file.name)
        for doc in texts:
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content
            if keyword.lower() in content.lower():
                # Show a snippet around the keyword
                idx = content.lower().find(keyword.lower())
                start = max(0, idx - 60)
                end = min(len(content), idx + 60)
                snippet = content[start:end].replace('\n', ' ')
                results.append(f"Page {page}: ...{snippet}...")
    if not results:
        return "No matches found."
    return "\n\n".join(results)

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["filename"] = os.path.basename(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    return texts, embeddings

with gr.Blocks() as demo:
    gr.Markdown(
        "# BondInsights: AI-Powered RAG Q&A and Due Diligence Report Generator"
    )
    
    with gr.Row():
        pdf_input = gr.Files(label="Upload PDF")
    with gr.Row():
        query_input = gr.Textbox(label="Enter your Question", placeholder="What is the credit rating of this bond?", lines=2)
    with gr.Row():
        query_btn = gr.Button("Ask a Question")
        # Add custom CSS for hover effect
        gr.HTML(
            """
            <style>
            button.svelte-1ipelgc:hover, button.svelte-1ipelgc:focus {
            background-color: #2563eb !important; /* Tailwind blue-600 */
            color: white !important;
            }
            </style>
            """
        )
        report_btn = gr.Button("Generate 1-2 Page Report")

    with gr.Row():
        search_input = gr.Textbox(label="Keyword or Phrase Search")
        
    with gr.Row():
        search_btn = gr.Button("Search Documents")

    search_output = gr.Textbox(label="Search Results", lines=10)
    output = gr.Textbox(label="Question Response", lines=20)
    report_output = gr.File(label="Download Report PDF")

    search_btn.click(fn=search_documents,inputs=[pdf_input, search_input],outputs=search_output)
    query_btn.click(fn=handle_query, inputs=[pdf_input, query_input], outputs=output)
    report_btn.click(fn=handle_report, inputs=pdf_input, outputs=report_output)

    

if __name__ == "__main__":
    demo.launch()