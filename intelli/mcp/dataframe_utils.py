"""
DataFrame MCP Utilities for Intelli

Tools for creating MCP servers that expose DataFrame operations.
Supports both Pandas and Polars DataFrames.
"""
import sys
from typing import List, Any, Dict, Optional, Union, TYPE_CHECKING

# Optional imports
PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    pass

POLARS_AVAILABLE = False
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    pass

# Type checking imports
if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# MCP import
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None


class BaseDataFrameMCPServerBuilder:
    """Base class for DataFrame MCP Server Builders"""

    def __init__(self, server_name: str, csv_file_path: str, initial_rows: Optional[int] = None, stateless_http: bool = False):
        if FastMCP is None:
            raise ImportError(
                "MCP server utilities require the 'mcp' module. "
                "Install it using 'pip install intelli[mcp]'."
            )
        self.mcp = FastMCP(server_name, stateless_http=stateless_http)
        self.tools: List[str] = []
        self.csv_file_path = csv_file_path
        self.initial_rows = initial_rows
        self.df: Any = None
        self._load_dataframe()

        # Register common DataFrame tools
        self._add_common_tools()

    def _load_dataframe(self):
        """Load DataFrame - implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_dataframe")

    def _add_common_tools(self):
        """Register common DataFrame tools with the MCP server."""
        
        @self.mcp.tool()
        def get_head(n: int = 5) -> str:
            """Returns the first n rows as JSON."""
            if self.df is None:
                return "Error: DataFrame not loaded."
            return self._df_to_json(self.df.head(n))
        self.tools.append(get_head.__name__)

        @self.mcp.tool()
        def get_schema() -> Dict[str, str]:
            """Returns column names and types."""
            if self.df is None:
                return {"error": "DataFrame not loaded."}
            return self._get_df_schema()
        self.tools.append(get_schema.__name__)

        @self.mcp.tool()
        def get_shape() -> Dict[str, int]:
            """Returns row and column counts."""
            if self.df is None:
                return {"error": "DataFrame not loaded."}
            rows, cols = self._get_df_shape()
            return {"rows": rows, "columns": cols}
        self.tools.append(get_shape.__name__)

        @self.mcp.tool()
        def select_columns(columns: List[str]) -> str:
            """Returns specific columns as JSON."""
            if self.df is None:
                return "Error: DataFrame not loaded."
            try:
                return self._select_df_columns(columns)
            except Exception as e:
                return f"Error selecting columns: {str(e)}"
        self.tools.append(select_columns.__name__)
        
        @self.mcp.tool()
        def filter_rows(column: str, operator: str, value: Any) -> str:
            """
            Filters rows by condition and returns as JSON.
            Operators: ==, !=, >, <, >=, <=, contains, in
            """
            if self.df is None:
                return "Error: DataFrame not loaded."
            try:
                return self._filter_df_rows(column, operator, value)
            except Exception as e:
                return f"Error filtering rows: {str(e)}"
        self.tools.append(filter_rows.__name__)

    def add_tool(self, func: callable) -> callable:
        """Add a custom tool that operates on self.df."""
        self.tools.append(func.__name__)
        return self.mcp.tool()(func)

    def run(self,
            transport: str = "stdio",
            mount_path: str = "/mcp",
            host: str = "0.0.0.0",
            port: int = 8000,
            print_info: bool = True) -> None:
        """Start the MCP server."""
        if print_info:
            self._print_server_info(transport, mount_path, host, port)
        
        if transport in ["http", "streamable-http"]:
            self.mcp.run(
                transport=transport,
                mount_path=mount_path
            )
        else:
            self.mcp.run(transport=transport)

    def _print_server_info(self, transport, mount_path, host, port):
        """Show server details."""
        tools_list = ", ".join(self.tools)
        print(f"Starting DataFrame MCP Server '{self.mcp.name}' using {self.__class__.__name__}...")
        print(f"Serving data from: {self.csv_file_path}")
        if self.initial_rows:
            print(f"Initially loading {self.initial_rows} rows.")
        print(f"Registered tools: {tools_list}")

        if transport in ["http", "streamable-http"]:
            print(f"Server will run at http://{host}:{port}")
            print(f"MCP endpoint URL: http://localhost:{port}{mount_path}")
            print("NOTE: Client must use arg_* parameter prefix (e.g., arg_n, arg_columns)")
        else:
            print(f"Server running with {transport} transport.")

    # Abstract methods to be implemented by subclasses
    def _df_to_json(self, df_subset: Any) -> str:
        raise NotImplementedError("Subclasses must implement _df_to_json")

    def _get_df_schema(self) -> Dict[str, str]:
        raise NotImplementedError("Subclasses must implement _get_df_schema")

    def _get_df_shape(self) -> tuple[int, int]:
        raise NotImplementedError("Subclasses must implement _get_df_shape")
    
    def _select_df_columns(self, columns: List[str]) -> str:
        raise NotImplementedError("Subclasses must implement _select_df_columns")

    def _filter_df_rows(self, column: str, operator: str, value: Any) -> str:
        raise NotImplementedError("Subclasses must implement _filter_df_rows")


class PandasMCPServerBuilder(BaseDataFrameMCPServerBuilder):
    """MCP server for Pandas DataFrames."""

    def __init__(self, server_name: str, csv_file_path: str, initial_rows: Optional[int] = None, stateless_http: bool = False):
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas library is not installed. Please install it with 'pip install pandas'.")
        super().__init__(server_name, csv_file_path, initial_rows, stateless_http)

    def _load_dataframe(self):
        try:
            self.df = pd.read_csv(self.csv_file_path, nrows=self.initial_rows)
            print(f"Pandas DataFrame loaded successfully from {self.csv_file_path} with shape {self.df.shape}.")
        except Exception as e:
            print(f"Error loading Pandas DataFrame: {e}")
            self.df = None

    def _df_to_json(self, df_subset: "pd.DataFrame") -> str:
        return df_subset.to_json(orient="records", indent=2)

    def _get_df_schema(self) -> Dict[str, str]:
        if self.df is None: return {}
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    def _get_df_shape(self) -> tuple[int, int]:
        if self.df is None: return (0,0)
        return self.df.shape
    
    def _select_df_columns(self, columns: List[str]) -> str:
        if self.df is None: return "Error: DataFrame not loaded."
        # Validate columns
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            return f"Error: The following columns were not found in the DataFrame: {', '.join(missing_cols)}"
        return self._df_to_json(self.df[columns])

    def _filter_df_rows(self, column: str, operator: str, value: Any) -> str:
        if self.df is None: return "Error: DataFrame not loaded."
        if column not in self.df.columns:
            return f"Error: Column '{column}' not found."

        # Convert value to the column's dtype
        try:
            col_dtype = self.df[column].dtype
            if pd.api.types.is_numeric_dtype(col_dtype):
                if operator == 'in':
                    if not isinstance(value, list):
                        return "Error: For 'in' operator, value must be a list."
                    value = [pd.to_numeric(v, errors='raise') for v in value]
                else:
                    value = pd.to_numeric(value, errors='raise')
            elif pd.api.types.is_string_dtype(col_dtype) and not isinstance(value, str) and operator != 'in':
                 value = str(value)
            elif operator == 'in' and isinstance(value, list):
                try:
                    value = [type(self.df[column].iloc[0])(v) for v in value]
                except:
                    pass # keep original if conversion fails

        except Exception as e:
            return f"Error converting value for filtering: {str(e)}. Ensure value type is compatible with column '{column}' ({col_dtype})."

        # Apply the filter
        filtered_df = None
        if operator == '==':
            filtered_df = self.df[self.df[column] == value]
        elif operator == '!=':
            filtered_df = self.df[self.df[column] != value]
        elif operator == '>':
            filtered_df = self.df[self.df[column] > value]
        elif operator == '<':
            filtered_df = self.df[self.df[column] < value]
        elif operator == '>=':
            filtered_df = self.df[self.df[column] >= value]
        elif operator == '<=':
            filtered_df = self.df[self.df[column] <= value]
        elif operator == 'contains':
            if not isinstance(value, str):
                return "Error: 'contains' operator requires a string value."
            # Ensure column is string type
            if not pd.api.types.is_string_dtype(self.df[column]):
                 filtered_df = self.df[self.df[column].astype(str).str.contains(value, case=False, na=False)]
            else:
                 filtered_df = self.df[self.df[column].str.contains(value, case=False, na=False)]
        elif operator == 'in':
            if not isinstance(value, list):
                 return "Error: 'in' operator requires a list value."
            filtered_df = self.df[self.df[column].isin(value)]
        else:
            return f"Error: Unsupported operator '{operator}'. Supported operators are '==', '!=', '>', '<', '>=', '<=', 'contains', 'in'."
        
        return self._df_to_json(filtered_df)


class PolarsMCPServerBuilder(BaseDataFrameMCPServerBuilder):
    """MCP server for Polars DataFrames."""

    def __init__(self, server_name: str, csv_file_path: str, initial_rows: Optional[int] = None, stateless_http: bool = False):
        if not POLARS_AVAILABLE:
            raise ImportError("Polars library is not installed. Please install it with 'pip install polars'.")
        super().__init__(server_name, csv_file_path, initial_rows, stateless_http)

    def _load_dataframe(self):
        try:
            self.df = pl.read_csv(self.csv_file_path, n_rows=self.initial_rows)
            print(f"Polars DataFrame loaded successfully from {self.csv_file_path} with shape {self.df.shape}.")
        except Exception as e:
            print(f"Error loading Polars DataFrame: {e}")
            self.df = None

    def _df_to_json(self, df_subset: "pl.DataFrame") -> str:
        import json
        return json.dumps(df_subset.to_dicts(), indent=2)

    def _get_df_schema(self) -> Dict[str, str]:
        if self.df is None: return {}
        return {col: str(dtype) for col, dtype in self.df.schema.items()}

    def _get_df_shape(self) -> tuple[int, int]:
        if self.df is None: return (0,0)
        return self.df.shape

    def _select_df_columns(self, columns: List[str]) -> str:
        if self.df is None: return "Error: DataFrame not loaded."
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            return f"Error: The following columns were not found in the DataFrame: {', '.join(missing_cols)}"
        return self._df_to_json(self.df.select(columns))

    def _filter_df_rows(self, column: str, operator: str, value: Any) -> str:
        if self.df is None: return "Error: DataFrame not loaded."
        if column not in self.df.columns:
            return f"Error: Column '{column}' not found."

        # Convert value to correct type for comparison
        try:
            col_dtype = self.df[column].dtype
            
            if operator == 'in':
                if not isinstance(value, list):
                    return "Error: For 'in' operator, value must be a list."
            elif isinstance(col_dtype, (pl.INTEGER_DTYPES, pl.FLOAT_DTYPES)):
                 value = float(value) if isinstance(col_dtype, pl.FLOAT_DTYPES) else int(value)
            elif col_dtype == pl.Boolean:
                 value = str(value).lower() in ['true', '1', 'yes']
            elif col_dtype == pl.Utf8 and not isinstance(value, str):
                 value = str(value)

        except Exception as e:
            return f"Error converting value for filtering: {str(e)}. Column '{column}' type is {col_dtype}."

        # Apply the filter
        col_expr = pl.col(column)
        
        if operator == '==':
            condition = col_expr == value
        elif operator == '!=':
            condition = col_expr != value
        elif operator == '>':
            condition = col_expr > value
        elif operator == '<':
            condition = col_expr < value
        elif operator == '>=':
            condition = col_expr >= value
        elif operator == '<=':
            condition = col_expr <= value
        elif operator == 'contains':
            if not isinstance(value, str):
                return "Error: 'contains' operator requires a string value."
            if self.df[column].dtype != pl.Utf8:
                 condition = col_expr.cast(pl.Utf8).str.contains(value, literal=False)
            else:
                 condition = col_expr.str.contains(value, literal=False)
        elif operator == 'in':
            if not isinstance(value, list):
                 return "Error: 'in' operator requires a list value."
            condition = col_expr.is_in(value)
        else:
            return f"Error: Unsupported operator '{operator}'. Supported operators are '==', '!=', '>', '<', '>=', '<=', 'contains', 'in'."
        
        filtered_df = self.df.filter(condition)
        return self._df_to_json(filtered_df)

# Test code (runs when script is executed directly)
if __name__ == '__main__':
    import os
    dummy_csv_path = "dummy_data_temp.csv"
    
    if PANDAS_AVAILABLE:
        print("--- Testing PandasMCPServerBuilder ---")
        try:
            data_pd = {'colA': [1, 2, 3, 4, 5], 'colB': ['x', 'y', 'z', 'x', 'y'], 'colC': [10.1, 20.2, 30.3, 40.4, 50.5]}
            pd.DataFrame(data_pd).to_csv(dummy_csv_path, index=False)
            
            pandas_server = PandasMCPServerBuilder("PandasTestServer", dummy_csv_path, initial_rows=3)
            if pandas_server.df is not None:
                print("Pandas Server DF head:")
                print(pandas_server.df.head())
                
                print("Schema:", pandas_server._get_df_schema())
                print("Shape:", pandas_server._get_df_shape())
                print("Head(2):", pandas_server._df_to_json(pandas_server.df.head(2)))
                print("Select ['colA', 'colC']:", pandas_server._select_df_columns(['colA', 'colC']))
                print("Filter colA > 2:", pandas_server._filter_df_rows('colA', '>', 2))
                print("Filter colB == 'x':", pandas_server._filter_df_rows('colB', '==', 'x'))
                print("Filter colB contains 'y':", pandas_server._filter_df_rows('colB', 'contains', 'y'))
                print("Filter colA in [1,3,5]:", pandas_server._filter_df_rows('colA', 'in', [1,3,5]))

            # Example server commands:
            # pandas_server.run(transport="stdio")
            # pandas_server.run(transport="streamable-http", port=8001)
        except ImportError:
            print("Pandas not available, skipping PandasMCPServerBuilder test.")
        except Exception as e:
            print(f"Error in Pandas test: {e}")
        finally:
            if os.path.exists(dummy_csv_path):
                os.remove(dummy_csv_path)

    if POLARS_AVAILABLE:
        print("--- Testing PolarsMCPServerBuilder ---")
        try:
            data_pl = {'colA': [1, 2, 3, 4, 5], 'colB': ['x', 'y', 'z', 'x', 'y'], 'colC': [10.1, 20.2, 30.3, 40.4, 50.5]}
            if not os.path.exists(dummy_csv_path):
                 pl.DataFrame(data_pl).write_csv(dummy_csv_path)

            polars_server = PolarsMCPServerBuilder("PolarsTestServer", dummy_csv_path, initial_rows=4)
            if polars_server.df is not None:
                print("Polars Server DF head:")
                print(polars_server.df.head())

                print("Schema:", polars_server._get_df_schema())
                print("Shape:", polars_server._get_df_shape())
                print("Head(2):", polars_server._df_to_json(polars_server.df.head(2)))
                print("Select ['colA', 'colC']:", polars_server._select_df_columns(['colA', 'colC']))
                print("Filter colA > 2:", polars_server._filter_df_rows('colA', '>', 2))
                print("Filter colB == 'x':", polars_server._filter_df_rows('colB', '==', 'x'))
                print("Filter colB contains 'y':", polars_server._filter_df_rows('colB', 'contains', 'y'))
                print("Filter colA in [1,3,5]:", polars_server._filter_df_rows('colA', 'in', [1,3,5]))

            # Example server commands:
            # polars_server.run(transport="stdio")
        except ImportError:
            print("Polars not available, skipping PolarsMCPServerBuilder test.")
        except Exception as e:
            print(f"Error in Polars test: {e}")
        finally:
            if os.path.exists(dummy_csv_path):
                os.remove(dummy_csv_path)
    
    if not PANDAS_AVAILABLE and not POLARS_AVAILABLE:
        print("Neither Pandas nor Polars is available. Skipping DataFrame server tests.") 