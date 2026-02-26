import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

const initialItems = [
  { id: "001", name: "Item One", status: "active", amount: "$1,200" },
  { id: "002", name: "Item Two", status: "pending", amount: "$840" },
  { id: "003", name: "Item Three", status: "completed", amount: "$3,500" },
]

export default function SaasInvoiceManagementPage() {
  const [search, setSearch] = useState("")
  const items = initialItems.filter(i => i.name.toLowerCase().includes(search.toLowerCase()))
  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Saas Invoice Management For Freelancers</h1>
        <Dialog>
          <DialogTrigger asChild><Button>+ New Item</Button></DialogTrigger>
          <DialogContent>
            <DialogHeader><DialogTitle>Create New Item</DialogTitle></DialogHeader>
            <div className="space-y-4 py-4">
              <Input placeholder="Name" />
              <Input placeholder="Amount" />
              <Button className="w-full">Save</Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>
      <Input placeholder="Search..." value={search} onChange={e => setSearch(e.target.value)} className="max-w-sm" />
      <Card>
        <CardContent className="p-0">
          <table className="w-full">
            <thead><tr className="border-b text-sm text-muted-foreground">
              <th className="p-4 text-left">ID</th>
              <th className="p-4 text-left">Name</th>
              <th className="p-4 text-left">Status</th>
              <th className="p-4 text-left">Amount</th>
              <th className="p-4"></th>
            </tr></thead>
            <tbody>{items.map(item => (
              <tr key={item.id} className="border-b hover:bg-muted/50">
                <td className="p-4 font-mono text-sm">{item.id}</td>
                <td className="p-4">{item.name}</td>
                <td className="p-4"><Badge variant={item.status === "active" ? "default" : item.status === "pending" ? "secondary" : "outline"}>{item.status}</Badge></td>
                <td className="p-4">{item.amount}</td>
                <td className="p-4">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild><Button variant="ghost" size="sm">⋯</Button></DropdownMenuTrigger>
                    <DropdownMenuContent><DropdownMenuItem>Edit</DropdownMenuItem><DropdownMenuItem>Delete</DropdownMenuItem></DropdownMenuContent>
                  </DropdownMenu>
                </td>
              </tr>
            ))}</tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  )
}
