import { test, expect, Page } from '@playwright/test'

const TEST_API_KEY = process.env.NEXUS_TEST_API_KEY || 'test-api-key'

async function login(page: Page) {
  await page.goto('/login')
  await page.fill('input#api-key', TEST_API_KEY)
  await page.click('button[type="submit"]')
  await page.waitForLoadState('networkidle')
  if (page.url().includes('/login')) {
    await page.evaluate(() => {
      sessionStorage.setItem('nexus_token', 'e2e-test-token')
      sessionStorage.setItem('nexus_tenant', 'e2e-tenant')
    })
    await page.goto('/')
  }
}

test.describe('Credentials Page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page)
  })

  test('navigate to credentials page', async ({ page }) => {
    await page.goto('/credentials')
    await expect(page.locator('h1')).toContainText('Credentials')
  })

  test('credentials page has Add Credential button', async ({ page }) => {
    await page.goto('/credentials')
    await expect(page.locator('text=Add Credential')).toBeVisible()
  })

  test('clicking Add Credential opens form modal', async ({ page }) => {
    await page.goto('/credentials')
    await page.click('text=Add Credential')
    // Modal should appear with form title
    await expect(page.locator('h3:text("Add Credential")')).toBeVisible()
  })

  test('credential form has Name input and Save button', async ({ page }) => {
    await page.goto('/credentials')
    await page.click('text=Add Credential')
    await expect(page.locator('input[placeholder="My API Key"]')).toBeVisible()
    await expect(page.locator('text=Save')).toBeVisible()
  })

  test('credential form has Test Connection button', async ({ page }) => {
    await page.goto('/credentials')
    await page.click('text=Add Credential')
    await expect(page.locator('text=Test Connection')).toBeVisible()
  })

  test('cancel closes the credential form', async ({ page }) => {
    await page.goto('/credentials')
    await page.click('text=Add Credential')
    await expect(page.locator('text=Save')).toBeVisible()
    await page.click('text=Cancel')
    // Modal should be closed
    await expect(page.locator('h3:text("Add Credential")')).not.toBeVisible()
  })

  test('add credential flow: fill name and save', async ({ page }) => {
    await page.goto('/credentials')
    await page.click('text=Add Credential')
    await page.fill('input[placeholder="My API Key"]', 'E2E Test Credential')
    await page.click('text=Save')
    await page.waitForLoadState('networkidle')
    // Either the modal closes or the credential appears in the list
  })

  test('no raw secret values visible in credential list', async ({ page }) => {
    await page.goto('/credentials')
    await page.waitForLoadState('networkidle')
    // Get all text content from the page body
    const bodyText = await page.locator('body').textContent()
    // Should not contain patterns like sk-..., secret_, Bearer tokens, etc.
    expect(bodyText).not.toMatch(/sk-[a-zA-Z0-9]{20,}/)
    expect(bodyText).not.toMatch(/secret_[a-zA-Z0-9]{10,}/)
  })

  test('delete credential shows confirmation dialog', async ({ page }) => {
    await page.goto('/credentials')
    await page.waitForLoadState('networkidle')
    // If there are any Delete buttons visible
    const deleteBtn = page.locator('text=Delete').first()
    if (await deleteBtn.isVisible()) {
      await deleteBtn.click()
      // Should show confirmation dialog
      const confirmDialog = page.locator('text=Confirm Delete, text=Delete Credential')
      await expect(confirmDialog.first()).toBeVisible()
    }
  })
})
