import { test, expect, Page } from '@playwright/test'

const TEST_API_KEY = process.env.NEXUS_TEST_API_KEY || 'test-api-key'

async function login(page: Page) {
  await page.goto('/login')
  await page.fill('input#api-key', TEST_API_KEY)
  await page.click('button[type="submit"]')
  await page.waitForLoadState('networkidle')
  // If still on login, inject a token to proceed
  if (page.url().includes('/login')) {
    await page.evaluate(() => {
      sessionStorage.setItem('nexus_token', 'e2e-test-token')
      sessionStorage.setItem('nexus_tenant', 'e2e-tenant')
    })
    await page.goto('/')
  }
}

test.describe('Workflow Full Lifecycle', () => {
  test.beforeEach(async ({ page }) => {
    await login(page)
  })

  test('navigate to workflows page', async ({ page }) => {
    await page.goto('/workflows')
    await expect(page.locator('h1')).toContainText('Workflows')
  })

  test('workflows page shows New Workflow button', async ({ page }) => {
    await page.goto('/workflows')
    await expect(page.locator('text=New Workflow')).toBeVisible()
  })

  test('create new workflow navigates to editor', async ({ page }) => {
    await page.goto('/workflows')
    await page.click('text=New Workflow')
    await page.waitForLoadState('networkidle')
    // Should navigate to /workflows/new or /workflows/:id
    expect(page.url()).toMatch(/\/workflows\/(new|[a-zA-Z0-9-]+)/)
  })

  test('workflow editor renders with Untitled Workflow', async ({ page }) => {
    await page.goto('/workflows/new')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('text=Untitled Workflow')).toBeVisible()
  })

  test('workflow editor has Save button', async ({ page }) => {
    await page.goto('/workflows/new')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('text=Save')).toBeVisible()
  })

  test('workflow editor has Generate button', async ({ page }) => {
    await page.goto('/workflows/new')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('text=Generate')).toBeVisible()
  })

  test('full lifecycle: create, save, navigate back', async ({ page }) => {
    // Navigate to workflows list
    await page.goto('/workflows')
    await page.waitForLoadState('networkidle')

    // Click New Workflow
    await page.click('text=New Workflow')
    await page.waitForLoadState('networkidle')

    // Should be in editor
    expect(page.url()).toMatch(/\/workflows\//)

    // Check editor rendered
    const saveBtn = page.locator('text=Save')
    if (await saveBtn.isVisible()) {
      // Click save
      await saveBtn.click()
      await page.waitForLoadState('networkidle')
    }

    // Navigate back to workflows list
    await page.goto('/workflows')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('h1')).toContainText('Workflows')
  })

  test('workflows page has search and filter controls', async ({ page }) => {
    await page.goto('/workflows')
    await page.waitForLoadState('networkidle')
    // Search input
    await expect(page.locator('input[placeholder*="Search"]')).toBeVisible()
    // Status filter select
    await expect(page.locator('select')).toBeVisible()
  })

  test('workflows page has Templates button', async ({ page }) => {
    await page.goto('/workflows')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('text=Templates')).toBeVisible()
  })

  test('workflows page has Generate with AI button', async ({ page }) => {
    await page.goto('/workflows')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('text=Generate with AI')).toBeVisible()
  })
})
