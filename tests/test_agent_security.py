from src.agent.groq_agent import TOOLS


def test_agent_has_no_arbitrary_sql_tool():
    names = {tool["function"]["name"] for tool in TOOLS}
    assert "execute_sql" not in names
    assert names == {
        "get_kpis",
        "regional_summary",
        "fuel_summary",
        "driver_age_summary",
        "find_policy",
        "top_anomalies",
        "anomaly_region_summary",
        "search_policy",
    }


def test_tool_schemas_are_closed():
    for tool in TOOLS:
        params = tool["function"]["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
