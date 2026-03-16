import argparse
import hashlib
import os


def write_text(path, content):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_reference_duanzi(md_text):
    items = []
    for line in (md_text or "").splitlines():
        s = line.strip()
        if not s.startswith("-"):
            continue
        s = s.lstrip("-").strip()
        if s.startswith("段子："):
            items.append(s)
    return items


def stable_pick(seed_text, items, idx):
    if not items:
        return ""
    h = hashlib.sha256((seed_text + "|" + str(idx)).encode("utf-8")).hexdigest()
    n = int(h[:8], 16)
    return items[n % len(items)]


def normalize_prefix(line):
    if not line:
        return "段子：  "
    if line.startswith("段子："):
        return "段子：  " + line.split("段子：", 1)[-1].lstrip()
    return "段子：  " + line.lstrip()


def generate_duanzi(topic, reference_items, idx):
    topic = (topic or "").strip()
    seed = topic or "默认"
    ref = stable_pick(seed, reference_items, idx)
    ref_body = ref.split("段子：", 1)[-1].strip() if ref else ""
    candidates = [
        f"甲：我决定把“{topic or '生活'}”做成可复用组件。乙：那稳定吗？甲：稳定，除了我。",
        f"甲：我给“{topic or '计划'}”做了版本管理。乙：有发布吗？甲：有，发布了一个更改日期的 README。",
        f"甲：我最近在研究“{topic or '效率'}”。乙：研究出什么了？甲：研究出我研究的时候最不效率。",
        f"甲：我把“{topic or '压力'}”放进回收站了。乙：清空了吗？甲：没，它提示：正在还原中。",
    ]
    body = stable_pick(seed + "|gen", candidates, idx) or candidates[idx % len(candidates)]
    if ref_body:
        body = body + " " + ref_body
    return normalize_prefix(body)


def build_md(topic):
    topic = (topic or "").strip() or "随便聊聊"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reference_path = os.path.abspath(os.path.join(base_dir, "..", "Reference", "段子参考.md"))
    reference_text = read_text(reference_path) if os.path.isfile(reference_path) else ""
    reference_items = parse_reference_duanzi(reference_text)

    duanzi_lines = [generate_duanzi(topic, reference_items, 0)]

    lines = []
    lines.append(f"# 段子：{topic}")
    lines.append("")
    lines.append("## 段子")
    lines.append("")
    lines.extend(duanzi_lines)
    lines.append("")
    lines.append("## 今日小剧场")
    lines.append("")
    lines.append("甲：我准备把生活过得更有条理。")
    lines.append("乙：怎么个条理法？")
    lines.append(f"甲：先给“{topic}”建个文件夹。")
    lines.append("乙：然后呢？")
    lines.append("甲：然后把烦恼都拖进去，点“删除”。")
    lines.append("乙：删除键好使吗？")
    lines.append("甲：不好使，它提示：权限不足——你还在意。")
    lines.append("")
    lines.append("## 结尾补刀")
    lines.append("")
    lines.append("- 你以为你在管理时间，其实时间在管理你。")
    lines.append(f"- 你以为“{topic}”会变简单，其实只是你变熟练了。")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="", help="主题，可选")
    parser.add_argument("--out", required=True, help="输出 md 路径")
    args = parser.parse_args()
    write_text(args.out, build_md(args.topic))
    print(args.out)


if __name__ == "__main__":
    main()
