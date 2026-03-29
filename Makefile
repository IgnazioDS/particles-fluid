# -----------------------------------------------------------------
# Particles Fluid  --  Next.js / TypeScript
# -----------------------------------------------------------------

.PHONY: install dev build start clean help

install:
	npm install

dev:
	npm run dev

build:
	npm run build

start:
	npm run start

clean:
	rm -rf .next node_modules/.cache

help:
	@echo ""
	@echo "  make install   -- install Node dependencies"
	@echo "  make dev       -- start dev server (localhost:3000)"
	@echo "  make build     -- production build"
	@echo "  make start     -- serve production build"
	@echo "  make clean     -- remove build cache"
	@echo ""
