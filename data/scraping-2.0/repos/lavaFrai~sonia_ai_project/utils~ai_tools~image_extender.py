import PIL.Image
import PIL.ImageDraw
import openai


class ImageExtender:
    @staticmethod
    async def extend_image(source_image, prepared_image_path, mask_image_path, target_size=1024):
        source_image = PIL.Image.open(source_image)

        width, height = source_image.size
        scale_factor = target_size / max(width, height)

        new_size = (int(width * scale_factor), int(height * scale_factor))
        source_image.thumbnail(new_size)

        square_image = PIL.Image.new('RGB', (target_size, target_size), (0, 0, 0))
        x = (square_image.width - source_image.width) // 2
        y = (square_image.height - source_image.height) // 2
        square_image.paste(source_image, (x, y))

        square_image.save(prepared_image_path)

        mask_draw = PIL.ImageDraw.Draw(source_image)
        mask_draw.rectangle((0, 0, new_size[0], new_size[1]), fill=(0, 0, 0))

        mask = PIL.Image.new('RGBA', (target_size, target_size), (255, 255, 255, 0))
        mask.paste(source_image, (x, y))
        mask.save(mask_image_path)

        response = await openai.Image.acreate_edit(
            image=open(prepared_image_path, "rb"),
            mask=open(mask_image_path, "rb"),
            prompt=" ",
            size=f"{target_size}x{target_size}"
        )
        return response["data"][0]["url"]
