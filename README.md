# TODO

Replace:

- `appName` with application name
- `appDescription` with application description.

Do `TODO` blocks and remove this block.

# appName

appDescription.

## Development

[//]: # (TODO: add links)

Install:

- python
- poetry

## Bootstrap CI/CD with GitHub Actions for Yandex Cloud

### Create secrets

Use [official doc](https://docs.github.com/ru/actions/security-guides/using-secrets-in-github-actions#creating-secrets-for-a-repository).

```properties
YC_SERVICE_ACCOUNT_KEY_FILE={content of sa-key.json}
YC_CONTAINER_REGISTRY_ID={cr_id}
```

- `cr_id` - id of container registry from Yandex Cloud
- `sa-key.json` - authorized key of service account from Yandex Cloud

### Workflow

Add this snapshot to `.github/workflows/docker.yml`:

```yaml
  - name: Login to Yandex Cloud Container Registry
    uses: yc-actions/yc-cr-login@v1
    with:
      yc-sa-json-credentials: ${{ secrets.YC_SERVICE_ACCOUNT_KEY_FILE }}

  - name: Tag and push to Yandex Cloud Container Registry
    run: |
      docker pull ${{ env.IMAGE }}
      docker tag ${{ env.IMAGE }} ${{ env.YCR_IMAGE }}
      docker push ${{ env.YCR_IMAGE }}
```
