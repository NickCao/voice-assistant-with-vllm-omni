FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .
USER 1001
ENTRYPOINT ["python", "src/agent.py"]
CMD ["start"]
