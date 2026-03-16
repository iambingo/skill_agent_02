import argparse
import hashlib
import os


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write_text(path, content):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def parse_reference_jokes(md_text):
    jokes = []
    for line in (md_text or "").splitlines():
        s = line.strip()
        if not s.startswith("-"):
            continue
        s = s.lstrip("-").strip()
        if s.startswith("冷笑话："):
            jokes.append(s)
    return jokes


def stable_pick(seed_text, items, idx):
    if not items:
        return ""
    h = hashlib.sha256((seed_text + "|" + str(idx)).encode("utf-8")).hexdigest()
    n = int(h[:8], 16)
    return items[n % len(items)]


def generate_cold_joke(topic, reference_jokes, idx):
    topic = (topic or "").strip()
    seed = topic or "默认"
    base = stable_pick(seed, reference_jokes, idx)
    base_tail = base.split("冷笑话：", 1)[-1].strip() if base else ""
    templates = [
        "我本来想聊{topic}，结果聊着聊着就变成了{topic}的“售后服务”。",
        "别人都说{topic}要靠天赋，我说不，我靠“天冷”。",
        "我问{topic}能不能更简单一点，它说可以：把复杂留给明天。",
        "我研究{topic}研究到一半，发现最难的部分叫“开始”。",
        "我给{topic}写了个总结：结论是，没结论。",
    ]
    t = stable_pick(seed + "|tpl", templates, idx) or templates[idx % len(templates)]
    if topic:
        core = t.format(topic=topic)
        if base_tail:
            core = core + " " + base_tail
        return "冷笑话：   " + core
    if base_tail:
        return "冷笑话：   " + base_tail
    return "冷笑话：   " + t.format(topic="人生")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="", help="主题，可选")
    parser.add_argument("--count", type=int, default=3, help="条数，默认 3")
    parser.add_argument("--out", required=True, help="输出 txt 路径")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    reference_path = os.path.abspath(os.path.join(base_dir, "..", "Reference", "冷笑话参考.md"))
    reference_text = read_text(reference_path) if os.path.isfile(reference_path) else ""
    reference_jokes = parse_reference_jokes(reference_text)

    count = max(1, int(args.count or 3))
    jokes = [generate_cold_joke(args.topic, reference_jokes, i) for i in range(count)]
    content = "\n".join(jokes).strip() + "\n"
    write_text(args.out, content)
    print(args.out)


if __name__ == "__main__":
    main()

