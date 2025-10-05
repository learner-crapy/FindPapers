import asyncio
import pdb
import os
import requests
from crawl4ai import AsyncWebCrawler
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy


async def get_web_page_content(url):
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        # Run the crawler on a URL
        result = await crawler.arun(url=url)

        # Print the extracted content
        return result.markdown


async def get_pdf_page_content(pdf_url):
    # Initialize the PDF crawler strategy
    pdf_crawler_strategy = PDFCrawlerStrategy()

    # PDFCrawlerStrategy is typically used in conjunction with PDFContentScrapingStrategy
    # The scraping strategy handles the actual PDF content extraction
    pdf_scraping_strategy = PDFContentScrapingStrategy()
    run_config = CrawlerRunConfig(scraping_strategy=pdf_scraping_strategy)

    async with AsyncWebCrawler(crawler_strategy=pdf_crawler_strategy) as crawler:
        # Example with a remote PDF URL
        # print(f"Attempting to process PDF: {pdf_url}")
        result = await crawler.arun(url=pdf_url, config=run_config)

        if result.success:
            # print(f"Successfully processed PDF: {result.url}")
            # print(f"Metadata Title: {result.metadata.get('title', 'N/A')}")
            # Further processing of result.markdown, result.media, etc.
            # would be done here, based on what PDFContentScrapingStrategy extracts.
            if result.markdown and hasattr(result.markdown, 'raw_markdown'):
                # print(f"Extracted text (first 200 chars): {result.markdown.raw_markdown}...")
                return result.markdown.raw_markdown
            else:
                # print("No markdown (text) content extracted.")
                return ''
        else:
            # print(f"Failed to process PDF: {result.error_message}")
            return ''

async def get_web_content(url, url_type):
    if url_type == "pdf":
        return await get_pdf_page_content(url)
    elif url_type == "web":
        return await get_web_page_content(url)
    else:
        raise Exception(f"Unknown url_type: {url_type}")


def download_pdf(link: str, save_dir: str = "downloads", file_name: str = None):
    """
    Download a PDF file from the given link and save it locally.

    Args:
        link (str): The URL to the PDF file (e.g., https://arxiv.org/pdf/2509.14034v1.pdf)
        save_dir (str): Directory to save the file. Default = "downloads"
        file_name (str): Optional custom file name (without extension)
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Derive file name
    if file_name is None:
        file_name = os.path.basename(link)
        if not file_name.endswith(".pdf"):
            file_name += ".pdf"

    save_path = os.path.join(save_dir, file_name)

    try:
        print(f"Downloading PDF from: {link}")
        response = requests.get(link, stream=True, timeout=30)
        response.raise_for_status()  # Raise error for HTTP 4xx/5xx

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB

        with open(save_path, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)

        print(f"Download complete: {save_path}")
        return save_path

    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(get_pdf_page_content('https://arxiv.org/pdf/2509.14034v1'))
    paper_link = "https://arxiv.org/pdf/2509.14034v1.pdf"
    save_dir = "downloads"
    download_pdf(paper_link, save_dir)
