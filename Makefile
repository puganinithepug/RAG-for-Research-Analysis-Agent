test:
	sudo docker compose up -d --build
	sudo docker compose run --rm rag python -m pytest tests/

