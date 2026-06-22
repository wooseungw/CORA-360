#!/bin/bash
# PanoLLaVA 평가 메트릭 의존성 설치 스크립트
# 모든 평가 메트릭(BLEU-4, METEOR, ROUGE-L, SPICE, CIDEr)을 사용하기 위한 설치

set -e

echo "=========================================="
echo "PanoLLaVA 평가 메트릭 설치"
echo "=========================================="

# 1. sacrebleu (BLEU-4)
echo ""
echo "[1/6] sacrebleu 설치 중 (BLEU-4)..."
pip install sacrebleu --quiet
echo "✓ sacrebleu 설치 완료"

# 2. NLTK (METEOR)
echo ""
echo "[2/6] NLTK 설치 중 (METEOR)..."
pip install nltk --quiet
python << 'EOF'
import nltk
try:
    nltk.data.find('corpora/wordnet')
    print("✓ NLTK wordnet 이미 설치됨")
except LookupError:
    print("  NLTK 데이터 다운로드 중...")
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    print("✓ NLTK 데이터 설치 완료")
EOF

# 3. rouge-score (ROUGE-L)
echo ""
echo "[3/6] rouge-score 설치 중 (ROUGE-L)..."
pip install rouge-score --quiet
echo "✓ rouge-score 설치 완료"

# 4. pycocoevalcap (SPICE, CIDEr)
echo ""
echo "[4/6] pycocoevalcap 설치 중 (SPICE, CIDEr)..."
pip install git+https://github.com/salaniz/pycocoevalcap.git --quiet 2>/dev/null || {
    echo "⚠️ GitHub 설치 실패. PyPI 대체 시도 중..."
    pip install pycocoevalcap --quiet || echo "⚠️ pycocoevalcap 설치 실패 (옵션)"
}
echo "✓ pycocoevalcap 설치 완료"

# 5. SentenceTransformer (SPICE 폴백)
echo ""
echo "[5/6] sentence-transformers 설치 중 (SPICE 폴백)..."
pip install sentence-transformers --quiet
echo "✓ sentence-transformers 설치 완료"

# 6. Java 확인 (SPICE 권장)
echo ""
echo "[6/6] Java 설치 여부 확인 중 (SPICE 선택사항)..."
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | grep version | awk '{print $3}' | tr -d '"')
    echo "✓ Java 설치됨 (버전: $JAVA_VERSION)"
    echo "  → SPICE 공식 구현 (StanfordCoreNLP) 사용 가능"
else
    echo "⚠️ Java 미설치"
    echo "  → SPICE는 의미적 유사도로 대체될 예정"
    echo ""
    echo "  Java 설치 방법:"
    echo "    Ubuntu/Debian: sudo apt-get install default-jdk"
    echo "    macOS: brew install openjdk"
    echo "    Windows: https://www.java.com/download/"
fi

echo ""
echo "=========================================="
echo "✅ 모든 평가 메트릭 설치 완료!"
echo "=========================================="
echo ""
echo "설치된 패키지:"
echo "  • sacrebleu       : BLEU-4 계산"
echo "  • NLTK            : METEOR 계산"
echo "  • rouge-score     : ROUGE-L 계산"
echo "  • pycocoevalcap   : SPICE, CIDEr 계산"
echo "  • sentence-trans  : SPICE 폴백 (의미 유사도)"
echo ""
echo "사용 방법:"
echo "  python scripts/eval.py --csv-input data/quic360/test.csv"
echo ""
echo "자세한 정보: docs/EVAL_METRICS_OFFICIAL_REPOS.md"
echo "=========================================="
