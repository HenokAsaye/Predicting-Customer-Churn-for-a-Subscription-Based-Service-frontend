"use client";

import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { User, CreditCard, Activity, PhoneIcon, Shield } from "lucide-react";
import type { CustomerData } from "@/lib/api";

interface CustomerFormProps {
  data: Partial<CustomerData>;
  onChange: (data: Partial<CustomerData>) => void;
}

export function CustomerForm({ data, onChange }: CustomerFormProps) {
  const updateField = <K extends keyof CustomerData>(
    field: K,
    value: CustomerData[K]
  ) => {
    onChange({ ...data, [field]: value } as Partial<CustomerData>);
  };

  return (
    <div className="space-y-8">

      <div className="space-y-4">
        <div className="flex items-center gap-2 text-primary">
          <User className="h-5 w-5" />
          <h3 className="font-semibold text-foreground">Demographics</h3>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="gender">Gender</Label>
            <Select
              value={(data.gender as string) || "Male"}
              onValueChange={(value) => updateField("gender", value as "Male" | "Female")}
            >
              <SelectTrigger id="gender" className="bg-secondary border-border">
                <SelectValue placeholder="Select gender" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Male">Male</SelectItem>
                <SelectItem value="Female">Female</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="senior">Senior Citizen</Label>
            <Select
              value={(data.SeniorCitizen ?? 0).toString()}
              onValueChange={(value) => updateField("SeniorCitizen", parseInt(value) as 0 | 1)}
            >
              <SelectTrigger id="senior" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">No</SelectItem>
                <SelectItem value="1">Yes</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="partner">Partner</Label>
            <Select
              value={(data.Partner as string) || "No"}
              onValueChange={(value) => updateField("Partner", value as "Yes" | "No")}
            >
              <SelectTrigger id="partner" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="dependents">Dependents</Label>
            <Select
              value={(data.Dependents as string) || "No"}
              onValueChange={(value) => updateField("Dependents", value as "Yes" | "No")}
            >
              <SelectTrigger id="dependents" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Account Info Section */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-primary">
          <CreditCard className="h-5 w-5" />
          <h3 className="font-semibold text-foreground">Account Information</h3>
        </div>
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="tenure">Tenure (months)</Label>
              <Input
                id="tenure"
                type="number"
                min={0}
                max={72}
                value={(data.tenure as number) || 0}
                onChange={(e) => updateField("tenure", parseInt(e.target.value) || 0)}
                className="bg-secondary border-border"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="monthly">Monthly Charges ($)</Label>
              <Input
                id="monthly"
                type="number"
                min={0}
                step={0.01}
                value={(data.MonthlyCharges as number) || 0}
                onChange={(e) => updateField("MonthlyCharges", parseFloat(e.target.value) || 0)}
                className="bg-secondary border-border"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="total">Total Charges ($)</Label>
              <Input
                id="total"
                type="number"
                min={0}
                step={0.01}
                value={(data.TotalCharges as number) || 0}
                onChange={(e) => updateField("TotalCharges", parseFloat(e.target.value) || 0)}
                className="bg-secondary border-border"
              />
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="contract">Contract</Label>
              <Select
                value={(data.Contract as string) || "Month-to-month"}
                onValueChange={(value) => updateField("Contract", value as any)}
              >
                <SelectTrigger id="contract" className="bg-secondary border-border">
                  <SelectValue placeholder="Select contract" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Month-to-month">Month-to-month</SelectItem>
                  <SelectItem value="One year">One year</SelectItem>
                  <SelectItem value="Two year">Two year</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="paperless">Paperless Billing</Label>
              <Select
                value={(data.PaperlessBilling as string) || "No"}
                onValueChange={(value) => updateField("PaperlessBilling", value as "Yes" | "No")}
              >
                <SelectTrigger id="paperless" className="bg-secondary border-border">
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Yes">Yes</SelectItem>
                  <SelectItem value="No">No</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="payment">Payment Method</Label>
              <Select
                value={(data.PaymentMethod as string) || "Credit card (automatic)"}
                onValueChange={(value) => updateField("PaymentMethod", value as any)}
              >
                <SelectTrigger id="payment" className="bg-secondary border-border">
                  <SelectValue placeholder="Select method" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Electronic check">Electronic check</SelectItem>
                  <SelectItem value="Mailed check">Mailed check</SelectItem>
                  <SelectItem value="Bank transfer (automatic)">Bank transfer (automatic)</SelectItem>
                  <SelectItem value="Credit card (automatic)">Credit card (automatic)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </div>

      {/* Phone & Internet Services */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-primary">
          <PhoneIcon className="h-5 w-5" />
          <h3 className="font-semibold text-foreground">Phone & Internet Services</h3>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="phone">Phone Service</Label>
            <Select
              value={(data.PhoneService as string) || "No"}
              onValueChange={(value) => updateField("PhoneService", value as "Yes" | "No")}
            >
              <SelectTrigger id="phone" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="multiple">Multiple Lines</Label>
            <Select
              value={(data.MultipleLines as string) || "No"}
              onValueChange={(value) => updateField("MultipleLines", value as any)}
            >
              <SelectTrigger id="multiple" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
                <SelectItem value="No phone service">No phone service</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="internet">Internet Service</Label>
            <Select
              value={(data.InternetService as string) || "DSL"}
              onValueChange={(value) => updateField("InternetService", value as any)}
            >
              <SelectTrigger id="internet" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="DSL">DSL</SelectItem>
                <SelectItem value="Fiber optic">Fiber optic</SelectItem>
                <SelectItem value="No">No</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Internet Add-ons */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-primary">
          <Shield className="h-5 w-5" />
          <h3 className="font-semibold text-foreground">Internet Add-ons</h3>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="security">Online Security</Label>
            <Select
              value={(data.OnlineSecurity as string) || "No internet service"}
              onValueChange={(value) => updateField("OnlineSecurity", value as any)}
            >
              <SelectTrigger id="security" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
                <SelectItem value="No internet service">No internet service</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="backup">Online Backup</Label>
            <Select
              value={(data.OnlineBackup as string) || "No internet service"}
              onValueChange={(value) => updateField("OnlineBackup", value as any)}
            >
              <SelectTrigger id="backup" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
                <SelectItem value="No internet service">No internet service</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="device">Device Protection</Label>
            <Select
              value={(data.DeviceProtection as string) || "No internet service"}
              onValueChange={(value) => updateField("DeviceProtection", value as any)}
            >
              <SelectTrigger id="device" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
                <SelectItem value="No internet service">No internet service</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="tech">Tech Support</Label>
            <Select
              value={(data.TechSupport as string) || "No internet service"}
              onValueChange={(value) => updateField("TechSupport", value as any)}
            >
              <SelectTrigger id="tech" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
                <SelectItem value="No internet service">No internet service</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="tv">Streaming TV</Label>
            <Select
              value={(data.StreamingTV as string) || "No internet service"}
              onValueChange={(value) => updateField("StreamingTV", value as any)}
            >
              <SelectTrigger id="tv" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
                <SelectItem value="No internet service">No internet service</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="movies">Streaming Movies</Label>
            <Select
              value={(data.StreamingMovies as string) || "No internet service"}
              onValueChange={(value) => updateField("StreamingMovies", value as any)}
            >
              <SelectTrigger id="movies" className="bg-secondary border-border">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Yes">Yes</SelectItem>
                <SelectItem value="No">No</SelectItem>
                <SelectItem value="No internet service">No internet service</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>
    </div>
  );
}
