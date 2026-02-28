# 第一阶段：使用 Hugo 镜像生成静态文件
FROM ghcr.io/gohugoio/hugo AS builder
WORKDIR /io.fynixoc.org
COPY  --chown=hugo:hugo . .
# 运行 Hugo 构建命令，生成静态文件到 
RUN ["hugo", "--minify", "--printPathWarnings", "--printI18nWarnings"] 

# 第二阶段：使用轻量的 Nginx 镜像进行托管
FROM nginx:alpine
# 清空 Nginx 默认路径下的所有内容，防止旧文件残留
RUN rm -rf /usr/share/nginx/html/*
# 将第一阶段生成的静态文件复制到 Nginx 的默认公开目录
COPY --from=builder /io.fynixoc.org/public /usr/share/nginx/html
# 暴露 80 端口
EXPOSE 80

