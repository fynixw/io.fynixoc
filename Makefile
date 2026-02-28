.PHONY: deploy

deploy:
	@echo "正在同步主题子模块..."
	git submodule update --init --recursive
	@echo "正在构建并启动容器..."
	docker-compose up -d --build
	@echo "部署完成！请访问 http://localhost:1313"

