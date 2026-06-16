import asyncio
import random
from typing import List, Dict

# ---------------------------------------------------------------------------
# Mô phỏng Vector DB với 10 tài liệu
# ---------------------------------------------------------------------------
MOCK_VECTOR_DB: Dict[str, str] = {
    "doc_001": "Chính sách đổi/trả hàng trong 30 ngày, cần hóa đơn, sản phẩm nguyên vẹn.",
    "doc_002": "Hướng dẫn đặt lại mật khẩu qua email, link hiệu lực 24h.",
    "doc_003": "Chính sách vận chuyển: miễn phí từ 500k, nội thành 1-2 ngày.",
    "doc_004": "Bảo hành điện tử 12 tháng, cần đăng ký online trong 7 ngày.",
    "doc_005": "Chương trình loyalty: tích điểm, hạng Bạc/Vàng/Kim Cương.",
    "doc_006": "Hỗ trợ kỹ thuật WiFi: restart, quên mạng, cập nhật firmware.",
    "doc_007": "Chính sách hoàn tiền hủy đơn: 100% chưa xử lý, 95% đang chuẩn bị.",
    "doc_008": "Hướng dẫn app mobile iOS 14+/Android 10+, thông báo đẩy, sinh trắc học.",
    "doc_009": "Bảo mật AES-256, không bán dữ liệu, quyền xóa tài khoản.",
    "doc_010": "Trả góp 0% lãi suất từ 3 triệu, kỳ hạn 3/6/12 tháng.",
}

KEYWORDS: Dict[str, List[str]] = {
    "doc_001": ["đổi", "trả", "hoàn", "sản phẩm", "hóa đơn", "30 ngày"],
    "doc_002": ["mật khẩu", "đặt lại", "quên", "reset", "email", "đăng nhập"],
    "doc_003": ["vận chuyển", "giao hàng", "ship", "phí", "nội thành", "ngoại tỉnh"],
    "doc_004": ["bảo hành", "điện tử", "kích hoạt", "sửa chữa", "hư hỏng"],
    "doc_005": ["điểm", "loyalty", "thân thiết", "hạng", "ưu đãi", "tích lũy"],
    "doc_006": ["wifi", "kết nối", "mạng", "internet", "router", "firmware"],
    "doc_007": ["hủy đơn", "hủy", "hoàn tiền", "refund", "cancel"],
    "doc_008": ["app", "ứng dụng", "mobile", "ios", "android", "thông báo"],
    "doc_009": ["bảo mật", "quyền riêng tư", "dữ liệu", "mã hóa", "cookies"],
    "doc_010": ["trả góp", "lãi suất", "thẻ tín dụng", "phê duyệt", "kỳ hạn"],
}


def mock_retrieve(question: str, top_k: int = 3) -> List[str]:
    """
    Mô phỏng vector search bằng keyword matching.
    Có xác suất nhỏ trả về sai để tạo failure cases thực tế.
    """
    question_lower = question.lower()
    scores: Dict[str, int] = {}
    for doc_id, keywords in KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in question_lower)
        if score > 0:
            scores[doc_id] = score

    # Sắp xếp theo điểm, lấy top_k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [doc_id for doc_id, _ in ranked[:top_k]]

    # Nếu không tìm được gì, trả về ngẫu nhiên (gây out-of-context case)
    if not top_ids:
        top_ids = random.sample(list(MOCK_VECTOR_DB.keys()), min(top_k, 3))

    # Thêm noise 20%: thỉnh thoảng đẩy 1 doc không liên quan vào
    if random.random() < 0.2 and len(top_ids) >= top_k:
        noise_doc = random.choice([d for d in MOCK_VECTOR_DB if d not in top_ids])
        top_ids[-1] = noise_doc

    return top_ids


def build_context(retrieved_ids: List[str]) -> str:
    """Ghép nội dung các tài liệu được retrieve."""
    return "\n\n".join(MOCK_VECTOR_DB.get(doc_id, "") for doc_id in retrieved_ids)


def mock_generate(question: str, context: str, retrieved_ids: List[str]) -> str:
    """
    Mô phỏng LLM generation dựa trên context.
    Trả về câu trả lời dựa trên nội dung retrieved.
    """
    if not retrieved_ids or not context.strip():
        return "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."

    templates = [
        f"Dựa trên tài liệu hệ thống, câu trả lời cho '{question}' là: {context[:200].strip()}...",
        f"Theo chính sách hiện hành: {context[:180].strip()}. Nếu cần hỗ trợ thêm, vui lòng liên hệ hotline.",
        f"Thông tin liên quan: {context[:200].strip()}. Mọi thắc mắc xin liên hệ bộ phận hỗ trợ.",
    ]
    return random.choice(templates)


class MainAgent:
    """
    RAG Agent mô phỏng với mock vector retrieval + mock LLM generation.
    Tích hợp đủ retrieved_ids để RetrievalEvaluator tính Hit Rate & MRR.
    """

    def __init__(self, version: str = "v1", top_k: int = 3):
        self.name = f"SupportAgent-{version}"
        self.version = version
        self.top_k = top_k

    async def query(self, question: str) -> Dict:
        """
        Pipeline RAG:
        1. Retrieve: keyword-based mock search → retrieved_ids
        2. Generate: mock LLM với context
        Returns dict đầy đủ cho BenchmarkRunner & RetrievalEvaluator.
        """
        # Giả lập độ trễ retrieval
        await asyncio.sleep(random.uniform(0.05, 0.2))

        # 1. Retrieval
        retrieved_ids = mock_retrieve(question, top_k=self.top_k)
        context = build_context(retrieved_ids)

        # Giả lập độ trễ LLM
        await asyncio.sleep(random.uniform(0.1, 0.3))

        # 2. Generation
        answer = mock_generate(question, context, retrieved_ids)

        tokens_used = len(question.split()) * 3 + len(answer.split()) * 2

        return {
            "answer": answer,
            "contexts": [MOCK_VECTOR_DB.get(doc_id, "") for doc_id in retrieved_ids],
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": "mock-gpt-4o-mini",
                "tokens_used": tokens_used,
                "sources": retrieved_ids,
                "version": self.version,
            },
        }


class MainAgentV2(MainAgent):
    """
    Agent V2: cải tiến top_k lên 5, giảm noise retrieval.
    Dùng để kiểm tra regression.
    """

    def __init__(self):
        super().__init__(version="v2", top_k=5)

    async def query(self, question: str) -> Dict:
        await asyncio.sleep(random.uniform(0.05, 0.15))
        retrieved_ids = mock_retrieve(question, top_k=self.top_k)

        # V2: xóa noise — giảm xác suất trả về doc ngẫu nhiên
        if random.random() < 0.05:
            noise_doc = random.choice(
                [d for d in MOCK_VECTOR_DB if d not in retrieved_ids]
            )
            retrieved_ids[-1] = noise_doc

        context = build_context(retrieved_ids)
        await asyncio.sleep(random.uniform(0.08, 0.2))
        answer = mock_generate(question, context, retrieved_ids)
        tokens_used = len(question.split()) * 3 + len(answer.split()) * 2

        return {
            "answer": answer,
            "contexts": [MOCK_VECTOR_DB.get(doc_id, "") for doc_id in retrieved_ids],
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": "mock-gpt-4o",
                "tokens_used": tokens_used,
                "sources": retrieved_ids,
                "version": "v2",
            },
        }


if __name__ == "__main__":
    async def test():
        agent = MainAgent()
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print("Answer:", resp["answer"])
        print("Retrieved IDs:", resp["retrieved_ids"])
    asyncio.run(test())
