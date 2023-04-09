import streamlit as st
import tag_model

# define css style for button
button_style = """
    <style>
    .stButton > button {
        width: 100%;
    }
    </style>
    """

st.markdown(button_style, unsafe_allow_html=True)

# define function to check input and generate tags
def check_input(tweet_input):
    if tweet_input:
        with st.spinner("Generating tags..."):
            tags = tag_model.get_tags(tweet_input)
        for i in tags:
            st.markdown("- " + i)
    else:
        st.write('Please enter a tweet')

# define function to clear input
def clear_input():
    st.session_state['tweet_input'] = ''


if __name__ == "__main__":
    # set page title and layout
    st.title('Tweet Tagging')

    tweet_input = st.text_area(label='Enter a tweet here to get tags:', value='', key='tweet_input')

    # create two columns for buttons
    col1, col2 = st.columns(2)

    # generate button
    with col1:
        button_gene = st.button('Generate')
        if button_gene:
            check_input(tweet_input)
    
    # clear input button
    with col2:
        button_clear = st.button('Clear',on_click=clear_input)