import streamlit as st
import pandas as pd
import base64
from io import BytesIO

def show_info(message, icon="ℹ️"):
    """
    Display an information message with an icon.
    
    Args:
        message (str): The message to display
        icon (str): The icon to show next to the message
    """
    st.markdown(f"<div style='display: flex; align-items: center;'>"
                f"<div style='font-size: 24px; margin-right: 10px;'>{icon}</div>"
                f"<div>{message}</div>"
                "</div>", unsafe_allow_html=True)

def download_dataframe(df, filename):
    """
    Create a download button for a dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to download
        filename (str): The filename for the downloaded file
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    st.markdown(href, unsafe_allow_html=True)

def format_currency(amount):
    """
    Format a value as currency.
    
    Args:
        amount (float): The amount to format
        
    Returns:
        str: Formatted currency string
    """
    return f"${amount:,.2f}"

def format_percentage(value):
    """
    Format a value as percentage.
    
    Args:
        value (float): The value to format
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value:.2f}%"

def format_number(value):
    """
    Format a value as a number with thousands separator.
    
    Args:
        value (float): The value to format
        
    Returns:
        str: Formatted number string
    """
    return f"{value:,}"
