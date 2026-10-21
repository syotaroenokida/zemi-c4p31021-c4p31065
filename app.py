from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)

# 日本語の音声認識を行うWhisperモデル
# 最初はbaseで動かす
model = whisper.load_model("base")

# アップロードされた音声を一時保存するフォルダ
UPLOAD_FOLDER = "tmp_uploaded_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        # ブラウザから送られてきた音声ファイルを取得
        if "audio" not in request.files:
            return jsonify({
                "error": "音声ファイルがありません"
            }), 400

        audio_file = request.files["audio"]

        # 一時ファイルとして保存
        audio_path = os.path.join(
            UPLOAD_FOLDER,
            "interview_audio.webm"
        )

        audio_file.save(audio_path)

        # Whisperで音声認識
        result = model.transcribe(
            audio_path,
            language="ja"
        )

        # 認識結果
        text = result["text"]

        # 一時ファイルを削除
        os.remove(audio_path)

        return jsonify({
            "text": text
        })

    except Exception as e:
        print("エラー:", e)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )