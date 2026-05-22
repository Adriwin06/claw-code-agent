from __future__ import annotations

from pathlib import Path

try:
    from rich.text import Text
except ModuleNotFoundError:  # pragma: no cover - Rich is provided by Textual.
    Text = None  # type: ignore[assignment]


_DEFAULT_BACKGROUND = (8, 16, 25)


def image_items_from_metadata(metadata: object) -> tuple[dict[str, object], ...]:
    if not isinstance(metadata, dict):
        return ()
    images = metadata.get('images')
    if not isinstance(images, list):
        return ()
    return tuple(dict(item) for item in images if isinstance(item, dict))


def render_terminal_image(
    path: Path,
    *,
    max_width: int = 64,
    max_height: int = 18,
    background: tuple[int, int, int] = _DEFAULT_BACKGROUND,
) -> object | None:
    if Text is None:
        return None
    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError:
        return None

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((max(1, max_width // 2), max_height), Image.Resampling.LANCZOS)
            rgba = image.convert('RGBA')
    except (OSError, ValueError):
        return None

    canvas = Image.new('RGBA', rgba.size, (*background, 255))
    canvas.alpha_composite(rgba)
    rgb = canvas.convert('RGB')
    width, height = rgb.size
    if width <= 0 or height <= 0:
        return None

    rendered = Text()
    pixels = rgb.load()
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            rendered.append(
                '  ',
                style=f'on rgb({red},{green},{blue})',
            )
        if y + 1 < height:
            rendered.append('\n')
    return rendered


def image_display_label(item: dict[str, object], *, index: int, total: int) -> str:
    path = str(
        item.get('source_url')
        or item.get('url')
        or item.get('path')
        or item.get('absolute_path')
        or '(unknown image)'
    )
    prefix = f'Image {index}/{total}' if total > 1 else 'Image'
    details: list[str] = []
    width = item.get('width')
    height = item.get('height')
    if isinstance(width, int) and isinstance(height, int):
        details.append(f'{width}x{height}')
    mime_type = item.get('mime_type')
    if isinstance(mime_type, str) and mime_type:
        details.append(mime_type)
    suffix = f' ({", ".join(details)})' if details else ''
    caption = item.get('caption')
    if isinstance(caption, str) and caption:
        return f'{prefix}: {caption} - {path}{suffix}'
    return f'{prefix}: {path}{suffix}'


def image_display_fallback(item: dict[str, object]) -> str:
    path = str(
        item.get('source_url')
        or item.get('url')
        or item.get('path')
        or item.get('absolute_path')
        or '(unknown image)'
    )
    uri = item.get('uri')
    lines = [
        'Inline thumbnail unavailable in this terminal/runtime.',
        'Install the TUI extra with Pillow, or rebuild the Docker image if using the launcher.',
        f'path={path}',
    ]
    if isinstance(uri, str) and uri:
        lines.append(f'uri={uri}')
    return '\n'.join(lines)
