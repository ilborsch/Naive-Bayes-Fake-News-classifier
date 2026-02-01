import pandas as pd
import os
from datetime import datetime


CSV_FILE = 'news_articles.csv'


def initialize_database():
    """
    Creates the CSV file if it doesn't exist
    """
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=['id', 'title', 'content', 'category', 'date_added', 'fake_label'])
        df.to_csv(CSV_FILE, index=False)


def read_all_articles(page: int = None, page_size: int = 10) -> pd.DataFrame:
    """
    Reads all articles with optional pagination
    """
    try:
        if not os.path.exists(CSV_FILE):
            initialize_database()

        df = pd.read_csv(CSV_FILE)

        if page is None:
            return df

        total_pages = (len(df) + page_size - 1) // page_size
        page = min(total_pages, max(page, 1))

        # Calculate start and end indices
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return df.iloc[start_idx:end_idx]

    except Exception as e:
        raise Exception(f"Failed to read articles: {str(e)}")


def create_article(title: str, content: str, category: str, fake_label: str) -> dict:
    """
    Adds a new article to the CSV file
    """
    try:
        df = read_all_articles()

        # Get next ID
        if len(df) == 0:
            next_id = 1
        else:
            next_id = int(df['id'].max()) + 1

        # Create new article
        new_article = {
            'id': next_id,
            'title': title.strip(),
            'content': content.strip(),
            'category': category,
            'date_added': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'fake_label': fake_label
        }

        # Append to dataframe
        df = pd.concat([df, pd.DataFrame([new_article])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

        return new_article

    except Exception as e:
        raise Exception(f"Failed to create article: {str(e)}")


def read_article_by_id(article_id: int) -> dict | None:
    """
    Read a single article by ID from the CSV
    """
    try:
        df = read_all_articles()
        article_df = df[df['id'] == article_id]

        if len(article_df) == 0:
            return None

        return article_df.iloc[0].to_dict()

    except Exception as e:
        raise Exception(f"Failed to read article {article_id}: {str(e)}")


def update_article(
        article_id: int,
        title: str = None,
        content: str = None,
        category: str = None,
) -> bool:
    """
    Updates an existing article in the CSV by its ID
    """
    try:
        df = read_all_articles()
        if article_id not in df['id'].values:
            return False

        if title is not None:
            df.loc[df['id'] == article_id, 'title'] = title.strip()

        if content is not None:
            df.loc[df['id'] == article_id, 'content'] = content.strip()

        if category is not None:
            df.loc[df['id'] == article_id, 'category'] = category

        df.to_csv(CSV_FILE, index=False)
        return True

    except Exception as e:
        raise Exception(f"Failed to update article {article_id}: {str(e)}")


def delete_article(article_id: int) -> bool:
    """
    Deletes article from the CSV by its ID
    """
    try:
        df = read_all_articles()

        if article_id not in df['id'].values:
            return False

        df = df[df['id'] != article_id]
        df.to_csv(CSV_FILE, index=False)
        return True

    except Exception as e:
        raise Exception(f"Failed to delete article {article_id}: {str(e)}")


def search_articles(query: str, column: str = 'title') -> pd.DataFrame:
    """
    Searches for articles by query & column
    """
    df = read_all_articles()

    if column not in df.columns:
        return pd.DataFrame()

    return df[df[column].str.contains(query, case=False, na=False)]



def get_total_pages(page_size: int = 10) -> int:
    """
    Helper function for UI. Returns total number of pages for pagination
    """
    df = read_all_articles()
    total_articles = len(df)
    return (total_articles + page_size - 1) // page_size


def get_total_articles() -> int:
    """
    Helper function for UI. Returns total number of articles
    """
    df = read_all_articles()
    return len(df)

