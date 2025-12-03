# import asyncio
# import edge_tts


# async def main():
#     voices = await edge_tts.list_voices()
#     for v in voices:
#         print(
#             v["ShortName"],
#             "-",
#             v["Gender"],
#             "Female",
#             v["Locale"],
#             "Thai",
#             v.get("FriendlyName", ""),
#             "Rose",
#         )

# asyncio.run(main())