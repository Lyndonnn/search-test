from PIL import Image
import numpy as np

from mmsearch_r1.utils.tools.offline_search import format_image_results, get_offline_index


def call_image_search(image_url: str):
    index = get_offline_index()
    if index is not None:
        results = index.search_image(image_url, topk=3)
        tool_returned_str, tool_returned_images, titles = format_image_results(results)
        tool_stat = {
            "success": True,
            "num_images": len(tool_returned_images),
            "backend": "offline_fvqa",
            "titles": titles,
        }
        return tool_returned_str, tool_returned_images, tool_stat

    print(
        "[Warning] You are currently using a *fake* implementation of the image search tool.\n"
        "This placeholder is intended for testing and does not perform real image retrieval.\n"
        "Set MMSEARCH_OFFLINE_PARQUET to a veRL parquet file to enable offline FVQA retrieval,\n"
        "or replace this function with logic that calls your own image search system or API."
    )

    tool_returned_images = []
    tool_returned_str = "[Image Search Results] The result of the image search consists of web page information related to the image from the user's original question. Each result includes the main image from the web page and its title, ranked in descending order of search relevance, as demonstrated below:\n"

    for i in range(3):
        dummy_img = Image.fromarray(np.full((64, 64, 3), fill_value=100 + i * 30, dtype=np.uint8))
        tool_returned_images.append(dummy_img)
        tool_returned_str += f"{i+1}. image: <|vision_start|><|image_pad|><|vision_end|>\ntitle: example webpage title {i+1}\n"

    tool_stat = {
        "success": True,
        "num_images": len(tool_returned_images),
        "backend": "fake",
        "titles": [f"Webpage Title {i + 1}" for i in range(len(tool_returned_images))],
    }
    return tool_returned_str, tool_returned_images, tool_stat
