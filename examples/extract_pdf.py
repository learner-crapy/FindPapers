import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy

async def main():
    # Initialize the PDF crawler strategy
    pdf_crawler_strategy = PDFCrawlerStrategy()

    # PDFCrawlerStrategy is typically used in conjunction with PDFContentScrapingStrategy
    # The scraping strategy handles the actual PDF content extraction
    pdf_scraping_strategy = PDFContentScrapingStrategy()
    run_config = CrawlerRunConfig(scraping_strategy=pdf_scraping_strategy)

    async with AsyncWebCrawler(crawler_strategy=pdf_crawler_strategy) as crawler:
        # Example with a remote PDF URL
        pdf_url = "https://aclanthology.org/2024.emnlp-industry.91.pdf" # A public PDF from arXiv

        print(f"Attempting to process PDF: {pdf_url}")
        result = await crawler.arun(url=pdf_url, config=run_config)

        if result.success:
            print(f"Successfully processed PDF: {result.url}")
            print(f"Metadata Title: {result.metadata.get('title', 'N/A')}")
            # Further processing of result.markdown, result.media, etc.
            # would be done here, based on what PDFContentScrapingStrategy extracts.
            if result.markdown and hasattr(result.markdown, 'raw_markdown'):
                print(f"Extracted text (first 200 chars): {result.markdown.raw_markdown}...")
            else:
                print("No markdown (text) content extracted.")
        else:
            print(f"Failed to process PDF: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
