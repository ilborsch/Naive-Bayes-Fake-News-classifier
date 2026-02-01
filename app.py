import streamlit as st
import pandas as pd
from database import *
from model import NaiveBayesModel, ModelNotLoadedError

CATEGORIES = ["Politics", "Sports", "Technology", "Entertainment"]


@st.cache_resource
def load_model():
    """
    Load model once and cache it.
    Avoids the bug where model is being reloaded each time streamlti updates it's UI
    """
    return NaiveBayesModel()


class NewsArticleApp:
    """Main application class for News Article Manager"""

    def __init__(self, model: NaiveBayesModel):
        """
        Initialize app with Naive Bayes model
        """
        self.model = model

        st.set_page_config(
            page_title="Fact Checked News",
            page_icon="📰",
            layout="wide"
        )
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'page' not in st.session_state:
            st.session_state.page = 'home'
        if 'article_id' not in st.session_state:
            st.session_state.article_id = None
        if 'search_query' not in st.session_state:
            st.session_state.search_query = ''
        if 'show_true_only' not in st.session_state:
            st.session_state.show_true_only = False

    @staticmethod
    def go_to_home():
        st.session_state.page = 'home'
        st.session_state.article_id = None

    @staticmethod
    def go_to_article(article_id):
        st.session_state.page = 'article'
        st.session_state.article_id = article_id

    @staticmethod
    def go_to_create():
        st.session_state.page = 'create'

    @staticmethod
    def get_label_color(label):
        if label == 'Fake Article':
            return 'red'
        elif label == 'Uncertain':
            return 'orange'
        else:
            return 'green'

    @staticmethod
    def shorten_content(content, max_length=50):
        if len(content) <= max_length:
            return content
        return content[:max_length] + '...'

    def render_search_controls(self):
        """
        Renders search bar and filters
        """
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            search = st.text_input(
                "🔍 Search by title or category",
                value=st.session_state.search_query,
                key='search_input'
            )
            st.session_state.search_query = search

        with col2:
            show_true_only = st.checkbox(
                "Include True only",
                value=st.session_state.show_true_only
            )
            st.session_state.show_true_only = show_true_only

        with col3:
            st.write("")  
            if st.button("➕ Create New Article", type="primary", use_container_width=True):
                self.go_to_create()
                st.rerun()

        return search, show_true_only

    def render_article_card(self, article):
        """
        Render a single article card with actions
        """
        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                # Title with colored label
                label_color = self.get_label_color(article['fake_label'])
                st.markdown(f"### {article['title']}")
                st.markdown(f":{label_color}[**{article['fake_label']}**]")
                st.write(f"**Category:** {article['category']} | **Date:** {article['date_added']}")
                st.write(self.shorten_content(article['content']))

            with col2:
                # Action buttons
                if st.button("📖 Read the article", key=f"read_{article['id']}", use_container_width=True):
                    self.go_to_article(article['id'])
                    st.rerun()

                if st.button("✏️ Update", key=f"update_{article['id']}", use_container_width=True):
                    st.session_state[f'update_mode_{article["id"]}'] = True

                if st.button("🗑️ Delete", key=f"delete_{article['id']}", use_container_width=True):
                    st.session_state[f'confirm_delete_{article["id"]}'] = True

            self.render_update_form(article)
            self.render_delete_confirmation(article)
            st.divider()

    def render_update_form(self, article):
        """Render inline update form for article"""
        if st.session_state.get(f'update_mode_{article["id"]}', False):
            with st.form(f"update_form_{article['id']}"):
                new_title = st.text_input("Title", value=article['title'], max_chars=200)
                new_content = st.text_area("Content", value=article['content'], height=150)
                new_category = st.selectbox(
                    "Category",
                    ["Politics", "Sports", "Technology", "Entertainment"],
                    index=["Politics", "Sports", "Technology", "Entertainment"].index(article['category'])
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    submitted = st.form_submit_button("💾 Save", use_container_width=True)
                with col_b:
                    cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)

                if cancelled:
                    st.session_state[f'update_mode_{article["id"]}'] = False
                    st.rerun()

                if submitted:
                    # Predict with model
                    combined_text = new_title + " " + new_content
                    predicted_label, confidence = self.model.predict_article_label(combined_text)

                    # Store pending update
                    st.session_state[f'pending_update_{article["id"]}'] = {
                        'article_id': article['id'],
                        'title': new_title,
                        'content': new_content,
                        'category': new_category,
                        'predicted_label': predicted_label,
                        'confidence': confidence * 100
                    }
                    st.session_state[f'show_update_prediction_{article["id"]}'] = True

            # Show prediction confirmation
            if st.session_state.get(f'show_update_prediction_{article["id"]}', False):
                pending = st.session_state[f'pending_update_{article["id"]}']
                label_color = self.get_label_color(pending['predicted_label'])

                st.divider()
                st.subheader("🤖 AI Prediction")
                st.markdown(f"Updated article is predicted as: :{label_color}[**{pending['predicted_label']}**]")
                st.write(f"Confidence: **{pending['confidence']:.1f}%**")
                st.warning("⚠️ Are you sure you want to update this article?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Yes, Update", key=f"confirm_update_{article['id']}", use_container_width=True,
                                 type="primary"):
                        if update_article(pending['article_id'], pending['title'], pending['content'],
                                          pending['category']):
                            st.success("✅ Article updated!")
                            st.session_state[f'update_mode_{article["id"]}'] = False
                            st.session_state[f'show_update_prediction_{article["id"]}'] = False
                            st.session_state[f'pending_update_{article["id"]}'] = None
                            st.rerun()

                with col2:
                    if st.button("❌ Cancel", key=f"cancel_update_{article['id']}", use_container_width=True):
                        st.session_state[f'show_update_prediction_{article["id"]}'] = False
                        st.session_state[f'pending_update_{article["id"]}'] = None
                        st.rerun()

    def render_delete_confirmation(self, article):
        """
        Render delete confirmation dialog
        """
        if st.session_state.get(f'confirm_delete_{article["id"]}', False):
            st.warning(f"⚠️ Are you sure you want to delete '{article['title']}'?")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Yes, Delete", key=f"confirm_yes_{article['id']}", use_container_width=True):
                    if delete_article(article['id']):
                        st.success("✅ Article deleted!")
                        st.session_state[f'confirm_delete_{article["id"]}'] = False
                        st.rerun()
            with col_b:
                if st.button("❌ Cancel", key=f"confirm_no_{article['id']}", use_container_width=True):
                    st.session_state[f'confirm_delete_{article["id"]}'] = False
                    st.rerun()

    def render_home_page(self):
        """
        Render home page with article list
        """
        st.title("📰 Fact Checked News")

        # Model status indicator
        if not self.model.is_model_loaded():
            st.error("❌ ML model not loaded properly!")

        # Search controls
        search, show_true_only = self.render_search_controls()
        st.divider()

        # Fetch and filter articles
        try:
            df = read_all_articles()

            # Apply search filter
            if search:
                title_matches = df['title'].str.contains(search, case=False, na=False)
                category_matches = df['category'].str.contains(search, case=False, na=False)
                df = df[title_matches | category_matches]

            # Apply "True only" filter
            if show_true_only:
                df = df[df['fake_label'] == 'True Article']

            if len(df) == 0:
                st.info("No articles found. Create your first article!")
            else:
                st.write(f"**Showing {len(df)} article(s)**")

                # Display articles
                for _, article in df.iterrows():
                    self.render_article_card(article)

        except Exception as e:
            st.error(f"Error loading articles: {str(e)}")


    def render_article_page(self):
        """
        Renders article detail page
        """
        if st.button("← Back to Home"):
            self.go_to_home()
            st.rerun()

        try:
            article = read_article_by_id(st.session_state.article_id)

            if article is None:
                st.error("Article not found")
                if st.button("← Back to Home"):
                    self.go_to_home()
                    st.rerun()
            else:
                # Display full article
                label_color = self.get_label_color(article['fake_label'])
                st.markdown(f":{label_color}[**{article['fake_label']}**]")
                st.title(article['title'])
                st.write(f"**Category:** {article['category']} | **Date Added:** {article['date_added']}")
                st.divider()
                st.write(article['content'])

        except Exception as e:
            st.error(f"Error loading article: {str(e)}")

    def render_create_form(self):
        """
        Renders create article form
        """
        with st.form("create_form"):
            title = st.text_input("Title*", max_chars=200)
            content = st.text_area("Content*", height=300)
            category = st.selectbox("Category*", CATEGORIES)

            col1, col2 = st.columns(2)

            with col1:
                submitted = st.form_submit_button("📝 Submit", use_container_width=True)

            with col2:
                cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)

            if cancelled:
                self.go_to_home()
                st.rerun()

            if submitted:
                if not title or not content:
                    st.error("❌ Please fill in all required fields")
                else:
                    # use model for prediction
                    combined_text = title + " " + content
                    predicted_label, confidence = self.model.predict_article_label(combined_text)

                    st.session_state['pending_article'] = {
                        'title': title,
                        'content': content,
                        'category': category,
                        'predicted_label': predicted_label,
                        'confidence': confidence * 100
                    }
                    st.session_state['show_prediction'] = True

    def render_prediction_confirmation(self):
        """
        Render model prediction confirmation dialog
        """
        if st.session_state.get('show_prediction', False):
            pending = st.session_state['pending_article']
            label_color = self.get_label_color(pending['predicted_label'])

            st.divider()
            st.subheader("🤖 AI Prediction")
            st.markdown(f"This article is predicted as: :{label_color}[**{pending['predicted_label']}**]")
            st.write(f"Confidence: **{pending['confidence']:.1f}%**")
            st.warning("⚠️ Are you sure you want to submit this article?")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Yes, Submit", use_container_width=True, type="primary"):
                    try:
                        create_article(
                            title=pending['title'],
                            content=pending['content'],
                            category=pending['category'],
                            fake_label=pending['predicted_label']
                        )
                        st.success("✅ Article created successfully!")
                        st.session_state['show_prediction'] = False
                        st.session_state['pending_article'] = None
                        self.go_to_home()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create article: {str(e)}")

            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state['show_prediction'] = False
                    st.session_state['pending_article'] = None
                    st.rerun()

    def render_create_page(self):
        st.title("➕ Create New Article")
        self.render_create_form()
        self.render_prediction_confirmation()

    def route(self):
        """
        Route to the correct page based on session state
        """
        if st.session_state.page == 'home':
            self.render_home_page()
        elif st.session_state.page == 'article':
            self.render_article_page()
        elif st.session_state.page == 'create':
            self.render_create_page()


if __name__ == "__main__":
    try:
        # Load model once (cached)
        nb_model = load_model()

        # Initialize and start app with model
        app = NewsArticleApp(model=nb_model)
        app.route()

    except ModelNotLoadedError as e:
        st.error(f"❌ {str(e)}")
        st.stop()

    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.stop()


