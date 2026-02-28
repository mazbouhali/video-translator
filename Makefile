# Arabic Video Translator - Makefile
# Usage: make help

.PHONY: help build run stop logs clean translate

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	docker compose build

run: ## Start translator in watch mode (background)
	docker compose up -d
	@echo "✓ Started! Drop videos in ./input/ folder"
	@echo "  View logs: make logs"
	@echo "  Stop: make stop"

up: run ## Alias for 'run'

start: run ## Alias for 'run'

stop: ## Stop translator
	docker compose down

down: stop ## Alias for 'stop'

logs: ## View live logs
	docker compose logs -f

restart: ## Restart translator
	docker compose restart

status: ## Show container status
	docker compose ps

shell: ## Open shell in container
	docker compose exec translator bash

# GPU detection
gpu-check: ## Check if GPU is available
	@docker compose run --rm translator python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Single file translation (without watch mode)
translate: ## Translate a single file: make translate VIDEO=path/to/video.mp4
ifndef VIDEO
	@echo "Usage: make translate VIDEO=path/to/video.mp4"
	@exit 1
endif
	@mkdir -p input output
	@cp "$(VIDEO)" input/
	docker compose run --rm translator python main.py --file "/app/input/$$(basename $(VIDEO))"
	@echo "✓ Done! Check ./output/ for results"

# Development
dev: ## Run in foreground (see all output)
	docker compose up

rebuild: ## Force rebuild and start
	docker compose up --build

# Cleanup
clean: ## Remove temp files and containers
	docker compose down -v
	rm -rf temp/*
	@echo "✓ Cleaned temp files"

clean-models: ## Remove downloaded models (will re-download)
	rm -rf models/*
	@echo "✓ Removed models (will download again on next run)"

clean-all: clean clean-models ## Remove everything except input/output
	docker rmi arabic-video-translator 2>/dev/null || true
	@echo "✓ Full cleanup complete"
