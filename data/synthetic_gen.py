"""
synthetic_gen.py — Golden Dataset Generator (SDG)

Chạy với OPENAI_API_KEY: gọi GPT để sinh QA tự động
Chạy không có API key: dùng static dataset 58 cases có sẵn
"""
import json
import asyncio
import random
import os
from typing import List, Dict

# Khởi tạo OpenAI client nếu có API key
try:
    from openai import AsyncOpenAI
    _api_key = os.environ.get("OPENAI_API_KEY", "")
    client = AsyncOpenAI(api_key=_api_key) if _api_key else None
except ImportError:
    client = None

# ---------------------------------------------------------------------------
# Tài liệu nguồn — mô phỏng Knowledge Base của hệ thống hỗ trợ kỹ thuật
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "doc_001": """
    [doc_001] Chính sách đổi/trả hàng
    Khách hàng có thể đổi hoặc trả sản phẩm trong vòng 30 ngày kể từ ngày mua.
    Sản phẩm phải còn nguyên tem nhãn, chưa qua sử dụng.
    Khách hàng cần xuất trình hóa đơn mua hàng khi đổi/trả.
    Các sản phẩm giảm giá trên 50% không áp dụng chính sách đổi/trả.
    Hoàn tiền sẽ được xử lý trong 5-7 ngày làm việc.
    """,
    "doc_002": """
    [doc_002] Hướng dẫn đặt lại mật khẩu
    Để đặt lại mật khẩu, truy cập trang đăng nhập và nhấn "Quên mật khẩu".
    Nhập địa chỉ email đã đăng ký, hệ thống sẽ gửi link xác nhận trong 2 phút.
    Link có hiệu lực trong 24 giờ. Sau khi đặt lại, phải đăng xuất toàn bộ thiết bị.
    Mật khẩu mới phải có ít nhất 8 ký tự, bao gồm chữ hoa, chữ thường và số.
    """,
    "doc_003": """
    [doc_003] Chính sách vận chuyển
    Miễn phí vận chuyển cho đơn hàng từ 500.000 VNĐ trở lên.
    Thời gian giao hàng nội thành: 1-2 ngày làm việc.
    Thời gian giao hàng ngoại tỉnh: 3-5 ngày làm việc.
    Vùng sâu vùng xa: 7-10 ngày làm việc.
    Phí vận chuyển tiêu chuẩn: 30.000 VNĐ. Giao hàng hỏa tốc: 60.000 VNĐ.
    """,
    "doc_004": """
    [doc_004] Quy định bảo hành sản phẩm điện tử
    Sản phẩm điện tử được bảo hành 12 tháng kể từ ngày mua.
    Bảo hành không áp dụng khi: sản phẩm bị hư hỏng do nước, rơi vỡ, tự sửa chữa.
    Để kích hoạt bảo hành, khách hàng cần đăng ký online trong 7 ngày sau mua.
    Trung tâm bảo hành mở cửa từ 8h-17h, thứ Hai đến thứ Sáu.
    """,
    "doc_005": """
    [doc_005] Chương trình khách hàng thân thiết (Loyalty Program)
    Khách hàng tích lũy 1 điểm cho mỗi 10.000 VNĐ mua hàng.
    Hạng Bạc: 100-499 điểm — giảm 5% mọi đơn hàng.
    Hạng Vàng: 500-999 điểm — giảm 10% + ưu tiên hỗ trợ.
    Hạng Kim Cương: 1000+ điểm — giảm 15% + giao hàng miễn phí mọi đơn.
    Điểm có giá trị sử dụng trong 12 tháng kể từ ngày tích lũy.
    """,
    "doc_006": """
    [doc_006] Hỗ trợ kỹ thuật — Kết nối WiFi
    Nếu thiết bị không kết nối được WiFi, thử các bước sau:
    1. Khởi động lại thiết bị và bộ định tuyến (router).
    2. Quên mạng WiFi hiện tại và kết nối lại từ đầu.
    3. Kiểm tra mật khẩu WiFi có đúng không.
    4. Đảm bảo firmware thiết bị đã được cập nhật phiên bản mới nhất.
    5. Nếu vẫn không được, liên hệ hotline 1800-xxxx để được hỗ trợ trực tiếp.
    """,
    "doc_007": """
    [doc_007] Chính sách hoàn tiền khi hủy đơn hàng
    Đơn hàng chưa được xử lý: hoàn 100% trong 24 giờ.
    Đơn hàng đang chuẩn bị: hoàn 95% (trừ 5% phí xử lý).
    Đơn hàng đã giao cho đơn vị vận chuyển: không thể hủy, phải chờ nhận hàng rồi đổi/trả.
    Hoàn tiền vào ví điện tử: 1-2 ngày làm việc.
    Hoàn tiền vào tài khoản ngân hàng: 5-7 ngày làm việc.
    """,
    "doc_008": """
    [doc_008] Hướng dẫn sử dụng ứng dụng di động
    Ứng dụng hỗ trợ iOS 14+ và Android 10+.
    Để cài đặt thông báo đẩy: Cài đặt > Thông báo > Bật tất cả.
    Tính năng thanh toán một chạm yêu cầu xác minh sinh trắc học (vân tay/Face ID).
    Dữ liệu được đồng bộ tự động mỗi 15 phút khi có kết nối internet.
    Phiên bản ứng dụng hiện tại: 3.2.1 (cập nhật ngày 01/05/2026).
    """,
    "doc_009": """
    [doc_009] Chính sách bảo mật và quyền riêng tư
    Dữ liệu khách hàng được mã hóa AES-256 và lưu trữ tại máy chủ đặt tại Việt Nam.
    Không bán thông tin cá nhân cho bên thứ ba vì mục đích thương mại.
    Khách hàng có quyền yêu cầu xóa tài khoản và toàn bộ dữ liệu cá nhân.
    Cookies chỉ được sử dụng để cải thiện trải nghiệm người dùng.
    Báo cáo vi phạm bảo mật qua email: security@company.com.
    """,
    "doc_010": """
    [doc_010] Quy định thanh toán trả góp
    Hỗ trợ trả góp 0% lãi suất cho đơn hàng từ 3.000.000 VNĐ trở lên.
    Kỳ hạn: 3, 6, 12 tháng tùy sản phẩm.
    Yêu cầu: CMND/CCCD, sao kê ngân hàng 3 tháng gần nhất.
    Thẻ tín dụng hỗ trợ: Visa, Mastercard, JCB.
    Thời gian phê duyệt: 15-30 phút trong giờ làm việc.
    """,
}

# ---------------------------------------------------------------------------
# Prompt tạo QA từ một tài liệu
# ---------------------------------------------------------------------------
SINGLE_DOC_PROMPT = """
Bạn là chuyên gia tạo bộ dữ liệu đánh giá RAG.

Từ tài liệu được cung cấp, hãy tạo {num_pairs} cặp QA đa dạng.

Yêu cầu:
- Bao gồm các loại: fact-check (dễ), reasoning (trung bình), adversarial (khó).
- Ít nhất 1 câu adversarial/edge-case (câu hỏi dễ gây hallucination).
- Chỉ sử dụng thông tin có trong tài liệu.
- Field "expected_retrieval_ids" PHẢI là [{doc_id}].
- Trả về JSON array thuần túy, không thêm markdown code block.

Schema:
[
  {{
    "question": "...",
    "expected_answer": "...",
    "context": "...",
    "expected_retrieval_ids": ["{doc_id}"],
    "metadata": {{
      "difficulty": "easy|medium|hard",
      "type": "fact-check|reasoning|adversarial"
    }}
  }}
]

Tài liệu:
{text}
"""

# ---------------------------------------------------------------------------
# Prompt tạo QA dạng hard cases yêu cầu nhiều tài liệu
# ---------------------------------------------------------------------------
MULTI_DOC_PROMPT = """
Bạn là chuyên gia tạo bộ dữ liệu đánh giá RAG.

Dựa trên CẢ HAI tài liệu sau, hãy tạo {num_pairs} câu hỏi phức tạp
đòi hỏi phải tổng hợp thông tin từ cả hai tài liệu để trả lời.

Yêu cầu:
- Câu hỏi phải liên quan đến cả hai tài liệu.
- Field "expected_retrieval_ids" PHẢI là [{doc_id_a}, {doc_id_b}].
- Trả về JSON array thuần túy, không thêm markdown code block.

Schema:
[
  {{
    "question": "...",
    "expected_answer": "...",
    "context": "...",
    "expected_retrieval_ids": ["{doc_id_a}", "{doc_id_b}"],
    "metadata": {{
      "difficulty": "hard",
      "type": "multi-hop"
    }}
  }}
]

Tài liệu A ({doc_id_a}):
{text_a}

Tài liệu B ({doc_id_b}):
{text_b}
"""

# ---------------------------------------------------------------------------
# Prompt tạo out-of-scope questions (kiểm tra hallucination)
# ---------------------------------------------------------------------------
OUT_OF_SCOPE_PROMPT = """
Bạn là chuyên gia tạo bộ dữ liệu kiểm thử AI.

Hãy tạo {num_pairs} câu hỏi NGOÀI PHẠM VI tài liệu được cung cấp.
Agent nên trả lời "Tôi không tìm thấy thông tin này trong tài liệu".

Trả về JSON array thuần túy, không thêm markdown code block.

Schema:
[
  {{
    "question": "...",
    "expected_answer": "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.",
    "context": "",
    "expected_retrieval_ids": [],
    "metadata": {{
      "difficulty": "hard",
      "type": "out-of-scope"
    }}
  }}
]

Tài liệu (nội dung KHÔNG liên quan đến câu hỏi):
{text}
"""


async def generate_from_single_doc(doc_id: str, text: str, num_pairs: int = 5) -> List[Dict]:
    """Tạo QA từ một tài liệu đơn."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia AI Evaluation. Luôn trả về JSON hợp lệ với key 'items'."},
            {"role": "user", "content": SINGLE_DOC_PROMPT.format(
                text=text, doc_id=doc_id, num_pairs=num_pairs
            )}
        ]
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        items = data.get("items", data.get("qa_pairs", []))
        if isinstance(items, list):
            return items
        # Nếu data là list trực tiếp
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"  ⚠️  Lỗi parse JSON cho {doc_id}: {e}")
        return []


async def generate_from_multi_doc(doc_id_a: str, text_a: str,
                                   doc_id_b: str, text_b: str,
                                   num_pairs: int = 2) -> List[Dict]:
    """Tạo QA đòi hỏi kết hợp hai tài liệu."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia AI Evaluation. Luôn trả về JSON hợp lệ với key 'items'."},
            {"role": "user", "content": MULTI_DOC_PROMPT.format(
                text_a=text_a, doc_id_a=doc_id_a,
                text_b=text_b, doc_id_b=doc_id_b,
                num_pairs=num_pairs
            )}
        ]
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        items = data.get("items", data.get("qa_pairs", []))
        return items if isinstance(items, list) else []
    except Exception as e:
        print(f"  ⚠️  Lỗi parse JSON multi-doc {doc_id_a}+{doc_id_b}: {e}")
        return []


async def generate_out_of_scope(text: str, num_pairs: int = 5) -> List[Dict]:
    """Tạo câu hỏi ngoài phạm vi tài liệu."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia AI Evaluation. Luôn trả về JSON hợp lệ với key 'items'."},
            {"role": "user", "content": OUT_OF_SCOPE_PROMPT.format(
                text=text, num_pairs=num_pairs
            )}
        ]
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        items = data.get("items", data.get("qa_pairs", []))
        return items if isinstance(items, list) else []
    except Exception as e:
        print(f"  ⚠️  Lỗi parse JSON out-of-scope: {e}")
        return []


STATIC_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")


def load_static_dataset() -> List[Dict]:
    """Đọc dataset tĩnh nếu đã tồn tại."""
    if os.path.exists(STATIC_DATASET_PATH):
        with open(STATIC_DATASET_PATH, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        return items
    return []


async def main():
    print("🏭 Bắt đầu tạo Golden Dataset (SDG)...")

    # Nếu không có API key → dùng dataset tĩnh
    if client is None:
        static = load_static_dataset()
        if static:
            print(f"ℹ️  Không có OPENAI_API_KEY. Dùng dataset tĩnh ({len(static)} cases) tại {STATIC_DATASET_PATH}")
            return
        else:
            print("❌ Không có OPENAI_API_KEY và không tìm thấy dataset tĩnh. Hãy set OPENAI_API_KEY.")
            return
    all_pairs: List[Dict] = []

    # --- Bước 1: Tạo 5 QA cho từng tài liệu đơn (10 docs × 5 = 50 cases) ---
    doc_ids = list(DOCUMENTS.keys())
    print(f"  📄 Tạo QA từ {len(doc_ids)} tài liệu đơn (×5 mỗi doc)...")
    tasks_single = [
        generate_from_single_doc(doc_id, DOCUMENTS[doc_id], num_pairs=5)
        for doc_id in doc_ids
    ]
    results_single = await asyncio.gather(*tasks_single)
    for batch in results_single:
        all_pairs.extend(batch)
    print(f"  ✅ Đã tạo {len(all_pairs)} cases từ tài liệu đơn.")

    # --- Bước 2: Tạo multi-hop cases từ cặp tài liệu ---
    doc_pairs = [
        ("doc_001", "doc_007"),  # đổi/trả & hoàn tiền
        ("doc_003", "doc_001"),  # vận chuyển & đổi/trả
        ("doc_005", "doc_010"),  # loyalty & trả góp
        ("doc_004", "doc_002"),  # bảo hành & mật khẩu
        ("doc_008", "doc_006"),  # app mobile & wifi
    ]
    print(f"  🔗 Tạo {len(doc_pairs)} cặp multi-hop cases...")
    tasks_multi = [
        generate_from_multi_doc(a, DOCUMENTS[a], b, DOCUMENTS[b], num_pairs=2)
        for a, b in doc_pairs
    ]
    results_multi = await asyncio.gather(*tasks_multi)
    for batch in results_multi:
        all_pairs.extend(batch)
    print(f"  ✅ Tổng sau multi-hop: {len(all_pairs)} cases.")

    # --- Bước 3: Tạo out-of-scope cases ---
    print("  🚫 Tạo out-of-scope / adversarial cases...")
    combined_text = list(DOCUMENTS.values())[0]  # dùng 1 doc làm context giả
    oos_pairs = await generate_out_of_scope(combined_text, num_pairs=5)
    all_pairs.extend(oos_pairs)
    print(f"  ✅ Tổng sau out-of-scope: {len(all_pairs)} cases.")

    # --- Shuffle để đa dạng hóa thứ tự ---
    random.shuffle(all_pairs)

    # --- Lưu ra file ---
    import os
    os.makedirs("data", exist_ok=True)
    output_path = "data/golden_set.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(all_pairs)} QA pairs to {output_path}")
    print("   Phân bố difficulty:")
    difficulty_count: Dict[str, int] = {}
    for p in all_pairs:
        d = p.get("metadata", {}).get("difficulty", "unknown")
        difficulty_count[d] = difficulty_count.get(d, 0) + 1
    for k, v in sorted(difficulty_count.items()):
        print(f"   - {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
