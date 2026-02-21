FROM node:20-alpine AS builder
WORKDIR /app

# Copy manifests first for layer caching
COPY package.json package-lock.json* ./
# Use npm ci when lock file is present (reproducible install), fall back to npm install
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
