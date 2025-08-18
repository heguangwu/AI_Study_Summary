from typing import Any
import httpx
import mcp
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import asyncio


BASE_URL = "https://kk6936hqgf.re.qweatherapi.com"
USER_AGENT = "weather-app/1.0"

mcp = FastMCP("weather")


@mcp.tool()
async def get_city_weather(city_code: str) -> str:
    """
    获取指定城市的天气预报

    Args:
        city_code: 城市编码
    """
    headers = {
        "User-Agent": USER_AGENT,
        "X-QW-Api-Key": os.getenv('API_KEY'),
        "Accept": "application/geo+json"
    }
    url = f"{BASE_URL}/v7/weather/3d?location={city_code}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            daily_data = response.json()['daily']
            forecasts = []
            for daily in daily_data:
                forecast = f"""
Date: {daily['fxDate']}
Temperature: {daily['tempMin']}°-{daily['tempMax']}°
Wind: {daily['windSpeedDay']} {daily['windDirDay']}
"""
                forecasts.append(forecast)

            return "\n---\n".join(forecasts)
        except Exception as e:
            return f"查询天气错误:{e}"

@mcp.tool()
async def get_location_weather(latitude: float, longitude: float) -> str:
    """
    获取指定位置的天气预报

    Args:
        latitude: 纬度
        longitude: 经度
    """
    location_id = await get_city_code(latitude, longitude)
    result = await get_city_weather(location_id)
    return result

@mcp.tool()
async def get_city_code(latitude: float, longitude: float) -> str:
    """
    根据经纬度获取城市编码

    Args:
        latitude: 纬度
        longitude: 经度
    """
    headers = {
        "User-Agent": USER_AGENT,
        "X-QW-Api-Key": os.getenv('API_KEY'),
        "Accept": "application/geo+json"
    }
    url = f"{BASE_URL}/geo/v2/city/lookup?location={longitude},{latitude}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            json_data = response.json()
            return json_data['location'][0]['id']
        except Exception as e:
            return f"获取城市编码错误：{e}"

def test():
    # result = asyncio.run(get_city_code(28.2282,112.9388))
    result = asyncio.run(get_city_weather("101250101"))
    print(result)


if __name__ == "__main__":
    load_dotenv()
    # Initialize and run the server
    mcp.run(transport='stdio')
