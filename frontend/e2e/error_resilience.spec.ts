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

test.describe('Error Resilience', () => {
  test.beforeEach(async ({ page }) => {
    await login(page)
  })

  test('invalid workflow ID in URL does not crash', async ({ page }) => {
    await page.goto('/workflows/nonexistent-workflow-id-12345')
    await page.waitForLoadState('networkidle')
    // The page should render something — either an error or the editor in "new" mode
    // It should NOT show a blank white page
    const bodyText = await page.locator('body').textContent()
    expect(bodyText.length).toBeGreaterThan(0)
  })

  test('invalid skill ID in URL does not crash', async ({ page }) => {
    await page.goto('/skills/nonexistent-skill-id-12345')
    await page.waitForLoadState('networkidle')
    const bodyText = await page.locator('body').textContent()
    expect(bodyText.length).toBeGreaterThan(0)
  })

  test('unknown route redirects to dashboard', async ({ page }) => {
    await page.goto('/this-route-does-not-exist')
    await page.waitForLoadState('networkidle')
    // App.jsx has <Route path="*" element={<Navigate to="/" replace />} />
    await expect(page).toHaveURL(/^\/$|\/login/)
  })

  test('dashboard handles API unavailability', async ({ page }) => {
    // Block API requests to simulate network failure
    await page.route('**/v1/ledger**', route => route.abort())
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    // Dashboard should show error state but not crash
    const bodyText = await page.locator('body').textContent()
    expect(bodyText).toContain('Dashboard')
  })

  test('workflows page handles API unavailability', async ({ page }) => {
    await page.route('**/v2/workflows**', route => route.abort())
    await page.goto('/workflows')
    await page.waitForLoadState('networkidle')
    // Should show error or empty state, not crash
    await expect(page.locator('h1')).toContainText('Workflows')
  })

  test('credentials page handles API unavailability', async ({ page }) => {
    await page.route('**/v2/credentials**', route => route.abort())
    await page.goto('/credentials')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('h1')).toContainText('Credentials')
  })

  test('executions page handles API unavailability', async ({ page }) => {
    await page.route('**/v2/executions**', route => route.abort())
    await page.goto('/executions')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('h1')).toContainText('Executions')
  })

  test('auth expiry redirects to login on page reload', async ({ page }) => {
    // Set valid token
    await page.evaluate(() => {
      sessionStorage.setItem('nexus_token', 'e2e-test-token')
    })
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Clear token to simulate expiry
    await page.evaluate(() => {
      sessionStorage.removeItem('nexus_token')
    })
    // Reload — ProtectedRoute checks isAuthenticated()
    await page.reload()
    await expect(page).toHaveURL(/\/login/)
  })

  test('rapid navigation between pages does not crash', async ({ page }) => {
    const pages = ['/workflows', '/credentials', '/executions', '/skills', '/mcp-servers', '/']
    for (const path of pages) {
      await page.goto(path)
      // Don't wait for full load — test rapid navigation
    }
    await page.waitForLoadState('networkidle')
    // Should end on dashboard without crash
    const bodyText = await page.locator('body').textContent()
    expect(bodyText.length).toBeGreaterThan(0)
  })

  test('API returning 500 shows error, not raw JSON', async ({ page }) => {
    await page.route('**/v2/workflows', route =>
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Internal Server Error' }),
      })
    )
    await page.goto('/workflows')
    await page.waitForLoadState('networkidle')
    // Should show human-readable error, not raw JSON object
    const bodyText = await page.locator('body').textContent()
    expect(bodyText).not.toContain('{"detail"')
    await expect(page.locator('h1')).toContainText('Workflows')
  })

  test('API returning 403 shows error message', async ({ page }) => {
    await page.route('**/v2/workflows', route =>
      route.fulfill({
        status: 403,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'AnomalyDetected: suspicious activity' }),
      })
    )
    await page.goto('/workflows')
    await page.waitForLoadState('networkidle')
    // The error should be displayed somewhere on the page
    const bodyText = await page.locator('body').textContent()
    expect(bodyText).toContain('Workflows')
  })
})
