"""
MCP DataFrame Server for Integration Testing.

This script runs an MCP server using either PandasMCPServerBuilder or PolarsMCPServerBuilder
to serve a sample CSV file via stdio transport. It checks for the availability of
PandaS or Polars and uses the first one found.

It expects the sample CSV to be located at ./data/sample_data.csv relative to this script.
"""
import os
import sys

# Adjust path to import from the parent directory (Intelli root)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add 'Intelli' to sys.path
intelli_root_path = parent_dir
sys.path.insert(0, intelli_root_path)

# Import from new location with fallback
from intelli.mcp.dataframe_utils import PandasMCPServerBuilder, PolarsMCPServerBuilder, PANDAS_AVAILABLE, POLARS_AVAILABLE

if __name__ == "__main__":
    # Determine the path to the sample CSV file
    # Assuming this server script is in test/integration/
    # and the data is in test/integration/data/
    csv_file_name = "sample_data.csv"
    csv_file_path = os.path.join(current_dir, "data", csv_file_name)

    if not os.path.exists(csv_file_path):
        print(f"Error: Sample CSV file not found at {csv_file_path}")
        print("Please ensure the 'sample_data.csv' is in the 'test/integration/data' directory.")
        sys.exit(1)

    server_builder = None
    server_type = ""

    if PANDAS_AVAILABLE:
        print("Pandas is available. Attempting to start Pandas DataFrame MCP Server.")
        try:
            server_builder = PandasMCPServerBuilder(
                server_name="PandasDataFrameTestServer",
                csv_file_path=csv_file_path
            )
            server_type = "Pandas"
        except Exception as e:
            print(f"Failed to initialize PandasMCPServerBuilder: {e}")
            server_builder = None # Ensure it's None if init fails
    
    if server_builder is None and POLARS_AVAILABLE:
        print("Pandas server failed or not available. Polars is available. Attempting to start Polars DataFrame MCP Server.")
        try:
            server_builder = PolarsMCPServerBuilder(
                server_name="PolarsDataFrameTestServer",
                csv_file_path=csv_file_path
            )
            server_type = "Polars"
        except Exception as e:
            print(f"Failed to initialize PolarsMCPServerBuilder: {e}")
            server_builder = None

    if server_builder and server_builder.df is not None:
        print(f"Successfully initialized {server_type} DataFrame MCP Server.")
        # Run the server with stdio transport for integration testing
        server_builder.run(transport="stdio", print_info=True)
    elif server_builder and server_builder.df is None:
        print(f"Initialized {server_type} DataFrame MCP Server, but DataFrame failed to load from {csv_file_path}.")
        print("Server will not run effectively. Please check CSV file and library installations.")
        sys.exit(1)
    else:
        print("Error: Neither Pandas nor Polars is available or server initialization failed.")
        print("Please install pandas or polars and ensure MCP is installed: pip install intelli[mcp] pandas polars")
        sys.exit(1) 