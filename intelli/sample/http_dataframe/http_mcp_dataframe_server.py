"""
HTTP MCP DataFrame Server

Serves CSV data using DataFrame operations over HTTP.
Supports Pandas or Polars, depending on what's installed.
"""
import os
import sys

# Add project root to path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
sys.path.insert(0, project_root_dir)

from utils.dataframe_mcp_utils import PandasMCPServerBuilder, PolarsMCPServerBuilder, PANDAS_AVAILABLE, POLARS_AVAILABLE

if __name__ == "__main__":
    # Get path to sample CSV
    csv_file_name = "sample_data.csv"
    csv_file_path = os.path.join(current_script_dir, csv_file_name)

    if not os.path.exists(csv_file_path):
        print(f"Error: Sample CSV file not found at {csv_file_path}")
        print(f"Please ensure '{csv_file_name}' exists in this directory.")
        sys.exit(1)

    server_builder = None
    server_type = ""
    server_name_prefix = "Http"

    # Try Pandas first
    if PANDAS_AVAILABLE:
        print("Using Pandas for DataFrame operations")
        try:
            server_builder = PandasMCPServerBuilder(
                server_name=f"{server_name_prefix}PandasDataFrameServer",
                csv_file_path=csv_file_path,
                stateless_http=True
            )
            server_type = "Pandas"
        except Exception as e:
            print(f"Failed to initialize PandasMCPServerBuilder: {e}")
            server_builder = None
    
    # Fall back to Polars if Pandas failed or isn't available
    if server_builder is None and POLARS_AVAILABLE:
        print("Using Polars for DataFrame operations")
        try:
            server_builder = PolarsMCPServerBuilder(
                server_name=f"{server_name_prefix}PolarsDataFrameServer",
                csv_file_path=csv_file_path,
                stateless_http=True
            )
            server_type = "Polars"
        except Exception as e:
            print(f"Failed to initialize PolarsMCPServerBuilder: {e}")
            server_builder = None

    if server_builder and server_builder.df is not None:
        print(f"Successfully initialized {server_type} DataFrame HTTP MCP Server.")
        
        # Configure HTTP server parameters
        mcp_path = "/mcp"  # Use the same path as the calculator example
        host = "0.0.0.0"
        port = 8000       # Must match client (and default Uvicorn port)
        
        # Start the server
        server_builder.run(
            transport="streamable-http", 
            mount_path=mcp_path,
            host=host, 
            port=port,
            print_info=True
        )
    elif server_builder and server_builder.df is None:
        print(f"Server initialized but DataFrame failed to load from {csv_file_path}.")
        print("Check that the CSV file is valid.")
        sys.exit(1)
    else:
        print("Error: Neither Pandas nor Polars is available.")
        print("Please install pandas or polars: pip install pandas polars")
        sys.exit(1) 