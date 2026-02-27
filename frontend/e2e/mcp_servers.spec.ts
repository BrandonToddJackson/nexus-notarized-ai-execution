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

test.describe('MCP Servers Page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page)
  })

  test('navigate to mcp-servers page', async ({ page }) => {
    await page.goto('/mcp-servers')
    await expect(page.locator('h1')).toContainText('MCP Servers')
  })

  test('mcp servers page has Add Server button', async ({ page }) => {
    await page.goto('/mcp-servers')
    await expect(page.locator('text=Add Server')).toBeVisible()
  })

  test('clicking Add Server opens modal with form', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.click('text=Add Server')
    await expect(page.locator('h3:text("Add MCP Server")')).toBeVisible()
  })

  test('add server form has Name input', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.click('text=Add Server')
    // The Name label and input should be visible
    await expect(page.locator('label:text("Name")')).toBeVisible()
  })

  test('add server form has transport radio options', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.click('text=Add Server')
    await expect(page.locator('label:text("stdio")')).toBeVisible()
    await expect(page.locator('label:text("sse")')).toBeVisible()
    await expect(page.locator('label:text("streamable_http")')).toBeVisible()
  })

  test('stdio transport shows Command and Arguments fields', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.click('text=Add Server')
    // stdio is default, so Command and Arguments should be visible
    await expect(page.locator('label:text("Command")')).toBeVisible()
    await expect(page.locator('label:text("Arguments")')).toBeVisible()
  })

  test('sse transport shows URL field', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.click('text=Add Server')
    // Switch to sse
    await page.click('label:text("sse")')
    await expect(page.locator('label:text("URL")')).toBeVisible()
    await expect(page.locator('input[placeholder="http://localhost:3000/mcp"]')).toBeVisible()
    // Command and Arguments should be hidden
    await expect(page.locator('label:text("Command")')).not.toBeVisible()
  })

  test('add button is disabled when name is empty', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.click('text=Add Server')
    // The submit "Add" button should be disabled
    const addBtns = page.locator('button:text("Add")')
    // Find the one inside the modal (not the header button)
    const modalAddBtn = page.locator('.fixed button:text("Add")')
    await expect(modalAddBtn).toBeDisabled()
  })

  test('cancel closes the add server modal', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.click('text=Add Server')
    await expect(page.locator('h3:text("Add MCP Server")')).toBeVisible()
    await page.click('text=Cancel')
    await expect(page.locator('h3:text("Add MCP Server")')).not.toBeVisible()
  })

  test('add HTTP server flow', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.click('text=Add Server')
    // Switch to sse transport
    await page.click('label:text("sse")')
    // Fill name
    const nameInput = page.locator('.fixed input').first()
    await nameInput.fill('E2E Test Server')
    // Fill URL
    await page.fill('input[placeholder="http://localhost:3000/mcp"]', 'http://localhost:9999/mcp')
    // The Add button should now be enabled
    const modalAddBtn = page.locator('.fixed button:text("Add")')
    await expect(modalAddBtn).toBeEnabled()
    await modalAddBtn.click()
    await page.waitForLoadState('networkidle')
  })

  test('server list shows column headers', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.waitForLoadState('networkidle')
    // If servers exist, column headers should be visible
    const hasServers = await page.locator('table').count()
    if (hasServers > 0) {
      await expect(page.locator('th:text("Name")')).toBeVisible()
      await expect(page.locator('th:text("Transport")')).toBeVisible()
      await expect(page.locator('th:text("Status")')).toBeVisible()
    }
  })

  test('remove server shows confirmation', async ({ page }) => {
    await page.goto('/mcp-servers')
    await page.waitForLoadState('networkidle')
    const removeBtns = page.locator('text=Remove')
    if (await removeBtns.count() > 0) {
      await removeBtns.first().click()
      await expect(page.locator('text=Remove MCP Server')).toBeVisible()
      // Cancel
      await page.click('text=Cancel')
    }
  })
})
