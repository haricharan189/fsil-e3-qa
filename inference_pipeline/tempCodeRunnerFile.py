                    elif from_label == "Organization Name" and to_label in ["Organization Role", "Organization Sub-Role"]:
                        org_uri = from_entity["uri"]
                        
                        if to_label == "Organization Role":
                            role_ns = self.org_role
                            role_uri = role_ns[self._clean_uri(to_entity["text"])]
                            
                            # Ensure role_uri is a URIRef
                            role_uri = URIRef(role_uri) if not isinstance(role_uri, URIRef) else role_uri
                            
                            # Add the role relationship
                            g.add((org_uri, self.isInstanceOf, role_uri))
                            
                            # Add sub-role relationships using role_subrole_pairs (filtering sub-roles for the role)
                            sub_roles = [sub_role_uri for role, sub_role_uri in role_subrole_pairs if role == role_uri]
                            
                            # If sub-roles exist, make the organization an instance of each sub-role
                            for sub_role_uri in sub_roles:
                                sub_role_uri = URIRef(sub_role_uri) if not isinstance(sub_role_uri, URIRef) else sub_role_uri
                                g.add((org_uri, self.isInstanceOf, sub_role_uri))
                        
                        elif to_label == "Organization Sub-Role":
                            # If it's an "Organization Sub-Role", find the parent role and link it
                            sub_role_ns = self.org_sub_role
                            sub_role_uri = sub_role_ns[self._clean_uri(to_entity["text"])]
                            
                            # Ensure sub_role_uri is a URIRef
                            sub_role_uri = URIRef(sub_role_uri) if not isinstance(sub_role_uri, URIRef) else sub_role_uri
                            
                            # Use the role-subrole mapping to find the corresponding parent role
                            parent_roles = [role_uri for role_uri, sub_role_uri in role_subrole_pairs if sub_role_uri == sub_role_uri]
                            
                            # Ensure parent_role_uri is a URIRef
                            for parent_role_uri in parent_roles:
                                parent_role_uri = URIRef(parent_role_uri) if not isinstance(parent_role_uri, URIRef) else parent_role_uri
                                g.add((org_uri, self.isInstanceOf, parent_role_uri))
                            
                            # Finally, add the sub-role itself
                            g.add((org_uri, self.isInstanceOf, sub_role_uri))
                        
                        # Add the role or sub-role to the org_roles mapping
                        org_roles[org_uri].add(role_uri)  # Add the role itself
                        if to_label == "Organization Sub-Role":
                            org_roles[org_uri].add(sub_role_uri)  # Add the sub-role as well